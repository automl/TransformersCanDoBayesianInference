
import torch
from torch import nn

class BarDistribution(nn.Module):
    def __init__(self, borders: torch.Tensor): # here borders should start with min and end with max, where all values lie in (min,max) and are sorted
        # sorted list of borders
        super().__init__()
        assert len(borders.shape) == 1
        #self.borders = borders
        self.register_buffer('borders', borders)
        #self.bucket_widths = self.borders[1:] - self.borders[:-1]
        self.register_buffer('bucket_widths', self.borders[1:] - self.borders[:-1])
        full_width = self.bucket_widths.sum()
        assert (full_width - (self.borders[-1] - self.borders[0])).abs() < 1e-4, f'diff: {full_width - (self.borders[-1] - self.borders[0])}'
        assert (torch.argsort(borders) == torch.arange(len(borders))).all(), "Please provide sorted borders!"
        self.num_bars = len(borders) - 1

    def map_to_bucket_idx(self, y):
        target_sample = torch.searchsorted(self.borders, y) - 1
        target_sample[y == self.borders[0]] = 0
        target_sample[y == self.borders[-1]] = self.num_bars - 1
        return target_sample

    def forward(self, logits, y): # gives the negative log density (the _loss_), y: T x B, logits: T x B x self.num_bars
        target_sample = self.map_to_bucket_idx(y)
        assert (target_sample >= 0).all() and (target_sample < self.num_bars).all(), f'y {y} not in support set for borders (min_y, max_y) {self.borders}'
        assert logits.shape[-1] == self.num_bars, f'{logits.shape[-1]} vs {self.num_bars}'

        bucket_log_probs = torch.log_softmax(logits, -1)
        scaled_bucket_log_probs = bucket_log_probs - torch.log(self.bucket_widths)

        return -scaled_bucket_log_probs.gather(-1,target_sample.unsqueeze(-1)).squeeze(-1)

    def mean(self, logits):
        bucket_means = self.borders[:-1] + self.bucket_widths/2
        p = torch.softmax(logits, -1)
        return p @ bucket_means

    def quantile(self, logits, center_prob=.682):
        logits_shape = logits.shape
        logits = logits.view(-1, logits.shape[-1])
        side_prob = (1-center_prob)/2
        probs = logits.softmax(-1)
        flipped_probs = probs.flip(-1)
        cumprobs = torch.cumsum(probs, -1)
        flipped_cumprobs = torch.cumsum(flipped_probs, -1)

        def find_lower_quantile(probs, cumprobs, side_prob, borders):
            idx = (torch.searchsorted(cumprobs, side_prob)).clamp(0, len(cumprobs)-1) # this might not do the right for outliers

            left_prob = cumprobs[idx-1]
            rest_prob = side_prob - left_prob
            left_border, right_border = borders[idx:idx+2]
            return left_border + (right_border-left_border)*rest_prob/probs[idx]

        results = []
        for p,cp,f_p,f_cp in zip(probs, cumprobs, flipped_probs, flipped_cumprobs):
            r = find_lower_quantile(p, cp, side_prob, self.borders), find_lower_quantile(f_p, f_cp, side_prob, self.borders.flip(0))
            results.append(r)

        return torch.tensor(results).reshape(*logits_shape[:-1],2)

    def mode(self, logits):
        mode_inds = logits.argmax(-1)
        bucket_means = self.borders[:-1] + self.bucket_widths/2
        return bucket_means[mode_inds]

    def ei(self, logits, best_f, maximize=True): # logits: evaluation_points x batch x feature_dim
        bucket_means = self.borders[:-1] + self.bucket_widths/2
        if maximize:
            bucket_contributions = torch.tensor(
                [max((bucket_max + max(bucket_min, best_f)) / 2 - best_f,0) for
                 bucket_min, bucket_max, bucket_mean in zip(self.borders[:-1], self.borders[1:], bucket_means)], dtype=logits.dtype, device=logits.device)
        else:
            bucket_contributions = torch.tensor(
                [-min((min(bucket_max,best_f) + bucket_min) / 2 - best_f,0) for # min on max instead of max on min, and compare min < instead of max >
                 bucket_min, bucket_max, bucket_mean in zip(self.borders[:-1], self.borders[1:], bucket_means)], dtype=logits.dtype, device=logits.device)
        p = torch.softmax(logits, -1)
        return p @ bucket_contributions


class FullSupportBarDistribution(BarDistribution):
    @staticmethod
    def halfnormal_with_p_weight_before(range_max,p=.5):
        s = range_max / torch.distributions.HalfNormal(torch.tensor(1.)).icdf(torch.tensor(p))
        return torch.distributions.HalfNormal(s)

    def forward(self, logits, y): # gives the negative log density (the _loss_), y: T x B, logits: T x B x self.num_bars
        assert self.num_bars > 1
        target_sample = self.map_to_bucket_idx(y)
        target_sample.clamp_(0,self.num_bars-1)
        assert logits.shape[-1] == self.num_bars

        bucket_log_probs = torch.log_softmax(logits, -1)
        scaled_bucket_log_probs = bucket_log_probs - torch.log(self.bucket_widths)
        #print(bucket_log_probs, logits.shape)
        log_probs = scaled_bucket_log_probs.gather(-1,target_sample.unsqueeze(-1)).squeeze(-1)

        side_normals = (self.halfnormal_with_p_weight_before(self.bucket_widths[0]), self.halfnormal_with_p_weight_before(self.bucket_widths[-1]))


        # TODO look over it again
        log_probs[target_sample == 0] += side_normals[0].log_prob((self.borders[1]-y[target_sample == 0]).clamp(min=.00000001)) + torch.log(self.bucket_widths[0])
        log_probs[target_sample == self.num_bars-1] += side_normals[1].log_prob(y[target_sample == self.num_bars-1]-self.borders[-2]) + torch.log(self.bucket_widths[-1])


        return -log_probs

    def mean(self, logits):
        bucket_means = self.borders[:-1] + self.bucket_widths / 2
        p = torch.softmax(logits, -1)
        side_normals = (self.halfnormal_with_p_weight_before(self.bucket_widths[0]),
                        self.halfnormal_with_p_weight_before(self.bucket_widths[-1]))
        bucket_means[0] = -side_normals[0].mean + self.borders[1]
        bucket_means[-1] = side_normals[1].mean + self.borders[-2]
        return p @ bucket_means



def get_bucket_limits(num_outputs:int, full_range:tuple=None, ys:torch.Tensor=None):
    assert (ys is not None) or (full_range is not None)
    if ys is not None:
        ys = ys.flatten()
        if len(ys) % num_outputs: ys = ys[:-(len(ys) % num_outputs)]
        print(f'Using {len(ys)} y evals to estimate {num_outputs} buckets. Cut off the last {len(ys) % num_outputs} ys.')
        ys_per_bucket = len(ys) // num_outputs
        if full_range is None:
            full_range = (ys.min(), ys.max())
        else:
            assert full_range[0] <= ys.min() and full_range[1] >= ys.max()
            full_range = torch.tensor(full_range)
        ys_sorted, ys_order = ys.sort(0)
        bucket_limits = (ys_sorted[ys_per_bucket-1::ys_per_bucket][:-1]+ys_sorted[ys_per_bucket::ys_per_bucket])/2
        print(full_range)
        bucket_limits = torch.cat([full_range[0].unsqueeze(0), bucket_limits, full_range[1].unsqueeze(0)],0)

    else:
        class_width = (full_range[1] - full_range[0]) / num_outputs
        bucket_limits = torch.cat([full_range[0] + torch.arange(num_outputs).float()*class_width, torch.tensor(full_range[1]).unsqueeze(0)], 0)

    assert len(bucket_limits) - 1 == num_outputs and full_range[0] == bucket_limits[0] and full_range[-1] == bucket_limits[-1]
    return bucket_limits




