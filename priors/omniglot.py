import math
import random
import torch
from torch.utils import data
from torchvision import transforms
import numpy as np

from datasets import omniglotNshot
import utils


def _compute_maxtranslations(single_image_tensor, dim, background):
    assert len(single_image_tensor.shape) == 2
    content_rows = ((single_image_tensor == background).all(dim=1 - dim) == False).nonzero()
    begin, end = content_rows[0], content_rows[-1]
    return torch.cat([-begin, single_image_tensor.shape[dim] - end - 1]).cpu().tolist()


def compute_maxtranslations_x_y(single_image_tensor, background):
    return _compute_maxtranslations(single_image_tensor, 1, background), _compute_maxtranslations(single_image_tensor,
                                                                                                  0, background)


def translate(img, trans_x, trans_y):
    return transforms.functional.affine(img.unsqueeze(0), angle=0.0, translate=[trans_x, trans_y], scale=1.0,
                                        interpolation=transforms.InterpolationMode.NEAREST, shear=[0.0, 0.0],
                                        fill=0.).squeeze(0)

def translate_omniglot(image_tensor, background=0.):
    flat_image_tensor = image_tensor.view(-1, *image_tensor.shape[-2:])
    for i, image in enumerate(flat_image_tensor):
        max_x, max_y = compute_maxtranslations_x_y(image, background)
        flat_image_tensor[i] = translate(image, random.randint(*max_x), random.randint(*max_y))
    return flat_image_tensor.view(*image_tensor.shape)


class DataLoader(data.DataLoader):
    def __init__(self, num_steps, batch_size, seq_len, num_features, num_outputs, num_classes_used=1200, fuse_x_y=False, train=True, translations=True, jonas_style=False):
        # TODO position before last is predictable by counting..
        utils.set_locals_in_self(locals())
        assert not fuse_x_y, 'So far don\' support fusing.'
        imgsz = math.isqrt(num_features)
        assert imgsz * imgsz == num_features
        assert ((seq_len-1) // num_outputs) * num_outputs == seq_len - 1
        if jonas_style:
            self.d = omniglotNshot.OmniglotNShotJonas('omniglot', batchsz=batch_size, n_way=num_outputs,
                                                 k_shot=((seq_len - 1) // num_outputs),
                                                 k_query=1, imgsz=imgsz)
        else:
            self.d = omniglotNshot.OmniglotNShot('omniglot', batchsz=batch_size, n_way=num_outputs,
                                                 k_shot=((seq_len - 1) // num_outputs),
                                                 k_query=1, imgsz=imgsz, num_train_classes_used=num_classes_used)


    def __len__(self):
        return self.num_steps

    def __iter__(self):
        # Eval at pos
        def t(x, y, x_q, y_q):
            x = np.concatenate([x,x_q[:,:1]], 1)
            y = np.concatenate([y,y_q[:,:1]], 1)
            y = torch.from_numpy(y).transpose(0, 1)
            target_y = y.clone().detach()
            target_y[:-1] = -100
            x = torch.from_numpy(x)
            if self.translations and self.train:
                x = translate_omniglot(x)
            image_tensor = x.view(*x.shape[:2], -1).transpose(0, 1), y
            return image_tensor, target_y

        return (t(*self.d.next(mode='train' if self.train else 'test')) for _ in range(self.num_steps))

    @torch.no_grad()
    def validate(self, finetuned_model, eval_pos=-1):
        finetuned_model.eval()
        device = next(iter(finetuned_model.parameters())).device

        if not hasattr(self, 't_dl'):
            self.t_dl = DataLoader(num_steps=self.num_steps, batch_size=self.batch_size, seq_len=self.seq_len,
                                   num_features=self.num_features, num_outputs=self.num_outputs, fuse_x_y=self.fuse_x_y,
                                   train=False)

        ps = []
        ys = []
        for x,y in self.t_dl:
            p = finetuned_model(tuple(e.to(device) for e in x), single_eval_pos=eval_pos)
            ps.append(p)
            ys.append(y)

        ps = torch.cat(ps,1)
        ys = torch.cat(ys,1)

        def acc(ps,ys):
            return (ps.argmax(-1)==ys.to(ps.device)).float().mean()

        a = acc(ps[eval_pos], ys[eval_pos]).cpu()
        return a
