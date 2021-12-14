import time
import functools
import random
import math
import traceback

import torch
from torch import nn
import gpytorch
from botorch.models import SingleTaskGP
from botorch.models.gp_regression import MIN_INFERRED_NOISE_LEVEL
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.priors.torch_priors import GammaPrior
from gpytorch.constraints import GreaterThan


from bar_distribution import BarDistribution
from utils import default_device
from .utils import get_batch_to_dataloader
from . import fast_gp

def get_model(x, y, hyperparameters: dict, sample=True):
    aug_batch_shape = SingleTaskGP(x,y.unsqueeze(-1))._aug_batch_shape
    noise_prior = GammaPrior(hyperparameters.get('noise_concentration',1.1), hyperparameters.get('noise_rate',0.05))
    noise_prior_mode = (noise_prior.concentration - 1) / noise_prior.rate
    likelihood = GaussianLikelihood(
        noise_prior=noise_prior,
        batch_shape=aug_batch_shape,
        noise_constraint=GreaterThan(
            MIN_INFERRED_NOISE_LEVEL,
            transform=None,
            initial_value=noise_prior_mode,
        ),
    )
    model = SingleTaskGP(x, y.unsqueeze(-1),
                         covar_module=gpytorch.kernels.ScaleKernel(
                            gpytorch.kernels.MaternKernel(
                                nu=hyperparameters.get('nu',2.5),
                                ard_num_dims=x.shape[-1],
                                batch_shape=aug_batch_shape,
                                lengthscale_prior=gpytorch.priors.GammaPrior(hyperparameters.get('lengthscale_concentration',3.0), hyperparameters.get('lengthscale_rate',6.0)),
                            ),
                            batch_shape=aug_batch_shape,
                            outputscale_prior=gpytorch.priors.GammaPrior(hyperparameters.get('outputscale_concentration',.5), hyperparameters.get('outputscale_rate',0.15)),
                        ), likelihood=likelihood)

    likelihood = model.likelihood
    if sample:
        sampled_model = model.pyro_sample_from_prior()
        return sampled_model, sampled_model.likelihood
    else:
        assert not(hyperparameters.get('sigmoid', False)) and not(hyperparameters.get('y_minmax_norm', False)), "Sigmoid and y_minmax_norm can only be used to sample models..."
        return model, likelihood


@torch.no_grad()
def get_batch(batch_size, seq_len, num_features, device=default_device, hyperparameters=None,
              batch_size_per_gp_sample=None, num_outputs=1,
              fix_to_range=None, equidistant_x=False):
    '''
    This function is very similar to the equivalent in .fast_gp. The only difference is that this function operates over
    a mixture of GP priors.
    :param batch_size:
    :param seq_len:
    :param num_features:
    :param device:
    :param hyperparameters:
    :param for_regression:
    :return:
    '''
    assert num_outputs == 1
    hyperparameters = hyperparameters or {}
    with gpytorch.settings.fast_computations(*hyperparameters.get('fast_computations',(True,True,True))):
        batch_size_per_gp_sample = (batch_size_per_gp_sample or max(batch_size // 10,1))
        assert batch_size % batch_size_per_gp_sample == 0

        total_num_candidates = batch_size*(2**(fix_to_range is not None))
        num_candidates = batch_size_per_gp_sample * (2**(fix_to_range is not None))
        if equidistant_x:
            assert num_features == 1
            x = torch.linspace(0,1.,seq_len).unsqueeze(0).repeat(total_num_candidates,1).unsqueeze(-1)
        else:
            x = torch.rand(total_num_candidates, seq_len, num_features, device=device)
        samples = []
        for i in range(0,total_num_candidates,num_candidates):
            model, likelihood = get_model(x[i:i+num_candidates], torch.zeros(num_candidates,x.shape[1]), hyperparameters)
            #print(model.covar_module.base_kernel.lengthscale)
            model.to(device)
            # trained_model = ExactGPModel(train_x, train_y, likelihood).cuda()
            # trained_model.eval()
            successful_sample = 0
            throwaway_share = 0.
            while successful_sample < 1:
                with gpytorch.settings.prior_mode(True):
                    d = model(x[i:i+num_candidates])
                    d = likelihood(d)
                    sample = d.sample() # bs_per_gp_s x T
                    if hyperparameters.get('y_minmax_norm'):
                        sample = ((sample - sample.min(1)[0]) / (sample.max(1)[0] - sample.min(1)[0]))
                    if hyperparameters.get('sigmoid'):
                        sample = sample.sigmoid()
                    if fix_to_range is None:
                        samples.append(sample.transpose(0, 1))
                        successful_sample = True
                        continue
                    smaller_mask = sample < fix_to_range[0]
                    larger_mask = sample >= fix_to_range[1]
                    in_range_mask = ~ (smaller_mask | larger_mask).any(1)
                    throwaway_share += (~in_range_mask[:batch_size_per_gp_sample]).sum()/batch_size_per_gp_sample
                    if in_range_mask.sum() < batch_size_per_gp_sample:
                        successful_sample -= 1
                        if successful_sample < 100:
                            print("Please change hyper-parameters (e.g. decrease outputscale_mean) it"
                                  "seems like the range is set to tight for your hyper-parameters.")
                        continue

                    x[i:i+batch_size_per_gp_sample] = x[i:i+num_candidates][in_range_mask][:batch_size_per_gp_sample]
                    sample = sample[in_range_mask][:batch_size_per_gp_sample]
                    samples.append(sample.transpose(0, 1))
                    successful_sample = True
        if random.random() < .01:
            print('throwaway share', throwaway_share/(batch_size//batch_size_per_gp_sample))

        #print(f'took {time.time() - start}')
        sample = torch.cat(samples, 1)
        x = x.view(-1,batch_size,seq_len,num_features)[0]
        # TODO think about enabling the line below
        #sample = sample - sample[0, :].unsqueeze(0).expand(*sample.shape)
        x = x.transpose(0,1)
        assert x.shape[:2] == sample.shape[:2]
        target_sample = sample
    return x, sample, target_sample # x.shape = (T,B,H)


class DataLoader(get_batch_to_dataloader(get_batch)):
    num_outputs = 1
    @torch.no_grad()
    def validate(self, model, step_size=1, start_pos=0):
        if isinstance(model.criterion, BarDistribution):
            (x,y), target_y = self.gbm(**self.get_batch_kwargs, fuse_x_y=self.fuse_x_y)
            model.eval()
            losses = []
            for eval_pos in range(start_pos, len(x), step_size):
                logits = model((x,y), single_eval_pos=eval_pos)
                means = model.criterion.mean(logits) # num_evals x batch_size
                mse = nn.MSELoss()
                losses.append(mse(means[0], target_y[eval_pos]))
            model.train()
            return torch.stack(losses)
        else:
            return 123.


@torch.enable_grad()
def get_fitted_model(x, y, hyperparameters, device):
    # fit the gaussian process
    model, likelihood = get_model(x,y,hyperparameters,sample=False)
    #print(model.covar_module.base_kernel.lengthscale)
    model.to(device)
    mll = ExactMarginalLogLikelihood(likelihood, model)
    model.train()
    fit_gpytorch_model(mll)
    #print(model.covar_module.base_kernel.lengthscale)
    return model, likelihood


evaluate = functools.partial(fast_gp.evaluate, get_model_on_device=get_fitted_model)

def get_mcmc_model(x, y, hyperparameters, device, num_samples, warmup_steps):
    from pyro.infer.mcmc import NUTS, MCMC
    import pyro
    x = x.to(device)
    y = y.to(device)
    model, likelihood = get_model(x, y, hyperparameters, sample=False)
    model.to(device)


    def pyro_model(x, y):
        sampled_model = model.pyro_sample_from_prior()
        _ = sampled_model.likelihood(sampled_model(x))
        return y

    nuts_kernel = NUTS(pyro_model, adapt_step_size=True)
    mcmc_run = MCMC(nuts_kernel, num_samples=num_samples, warmup_steps=warmup_steps)
    #print(x.shape)
    mcmc_run.run(x, y)
    model.pyro_load_from_samples(mcmc_run.get_samples())
    model.eval()
    # test_x = torch.linspace(0, 1, 101).unsqueeze(-1)
    # test_y = torch.sin(test_x * (2 * math.pi))
    # expanded_test_x = test_x.unsqueeze(0).repeat(num_samples, 1, 1)
    # output = model(expanded_test_x)
    #print(x.shape)
    return model, likelihood
    # output = model(x[-1].unsqueeze(1).repeat(1, num_samples 1))
    # return output.mean




def get_mean_logdensity(dists, x: torch.Tensor, full_range=None):
    means = torch.cat([d.mean.squeeze() for d in dists], 0)
    vars = torch.cat([d.variance.squeeze() for d in dists], 0)
    assert len(means.shape) == 1 and len(vars.shape) == 1
    dist = torch.distributions.Normal(means, vars.sqrt())
    #logprobs = torch.cat([d.log_prob(x) for d in dists], 0)
    logprobs = dist.log_prob(x)
    if full_range is not None:
        used_weight = 1. - (dist.cdf(torch.tensor(full_range[0])) + (1.-dist.cdf(torch.tensor(full_range[1]))))
        if torch.isinf(-torch.log(used_weight)).any() or torch.isinf(torch.log(used_weight)).any():
            print('factor is inf', -torch.log(used_weight))
        logprobs -= torch.log(used_weight)
    assert len(logprobs.shape) == 1
    #print(logprobs)
    return torch.logsumexp(logprobs, 0) - math.log(len(logprobs))


def evaluate_(x, y, y_non_noisy, hyperparameters=None, device=default_device, num_samples=100, warmup_steps=300,
              full_range=None, min_seq_len=0, use_likelihood=False):
    with gpytorch.settings.fast_computations(*hyperparameters.get('fast_computations',(True,True,True))), gpytorch.settings.fast_pred_var(False):
        x = x.to(device)
        y = y.to(device)
        start_time = time.time()
        losses_after_t = [.0] if min_seq_len == 0 else []
        all_losses = []

        for t in range(max(min_seq_len,1), len(x)):
            #print('Timestep', t)
            loss_sum = 0.
            step_losses = []
            start_step = time.time()
            for b_i in range(x.shape[1]):
                done = 0
                while done < 1:
                    try:
                        model, likelihood = get_mcmc_model(x[:t, b_i], y[:t, b_i], hyperparameters, device, num_samples=num_samples, warmup_steps=warmup_steps)
                        model.eval()

                        with torch.no_grad():
                            dists = model(x[t, b_i, :].unsqueeze(
                                0))  # TODO check what is going on here! Does the GP interpret the input wrong?
                            if use_likelihood:
                                dists = likelihood(dists)
                            l = -get_mean_logdensity([dists], y[t, b_i], full_range)
                        done = 1
                    except Exception as e:
                        done -= 1
                        print('Trying again..')
                        print(traceback.format_exc())
                        print(e)
                    finally:
                        if done < -10:
                            print('Too many retries...')
                            exit()

                step_losses.append(l.item())
                #print('loss',l.item())
                print(f'current average loss at step {t} is {sum(step_losses)/len(step_losses)} with {(time.time()-start_step)/len(step_losses)} s per eval.')
                loss_sum += l

            loss_sum /= x.shape[1]
            all_losses.append(step_losses)
            print(f'loss after step {t} is {loss_sum}')
            losses_after_t.append(loss_sum)
            print(f'losses so far {torch.tensor(losses_after_t)}')
        return torch.tensor(losses_after_t), time.time() - start_time, all_losses





if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--seq_len', type=int)
    parser.add_argument('--min_seq_len', type=int, default=0)
    parser.add_argument('--warmup_steps', type=int)
    parser.add_argument('--num_samples', type=int)
    parser.add_argument('--min_y', type=int)
    parser.add_argument('--max_y', type=int)
    parser.add_argument('--dim', type=int, default=1)
    parser.add_argument('--use_likelihood', default=True, type=bool)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--outputscale_concentraion', default=2., type=float)
    parser.add_argument('--noise_concentration', default=1.1, type=float)
    parser.add_argument('--noise_rate', default=.05, type=float)

    args = parser.parse_args()

    print('min_y:', args.min_y)
    full_range = (None if args.min_y is None else (args.min_y,args.max_y))

    hps = {'outputscale_concentration': args.outputscale_concentraion, 'noise_concentration': args.noise_concentration,
           'noise_rate': args.noise_rate, 'fast_computations': (False,False,False)}
    x, y, _ = get_batch(args.batch_size, args.seq_len, args.dim, fix_to_range=full_range, hyperparameters=hps)
    print('RESULT:', evaluate_(x, y, y, device=args.device, warmup_steps=args.warmup_steps,
                               num_samples=args.num_samples, full_range=full_range, min_seq_len=args.min_seq_len,
                               hyperparameters=hps, use_likelihood=args.use_likelihood))


