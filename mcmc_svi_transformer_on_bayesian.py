import scipy.stats as st
from train import Losses
import argparse

import os

from tqdm import tqdm
import time

import torch
import numpy as np
import matplotlib.pyplot as plt

import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
import torch.nn as nn
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.infer import SVI, Trace_ELBO, Predictive, MCMC, NUTS
from pyro import infer
import matplotlib.gridspec as gridspec
import os.path
import glob
from train import train, get_weighted_single_eval_pos_sampler
import priors
import encoders
from pyro.infer import SVGD, RBFSteinKernel

class CausalModel(PyroModule):
    def __init__(self, model_spec, device='cuda'):
        super().__init__()

        self.device = device
        self.num_features = model_spec['num_features']

        mu, sigma = torch.tensor([0.0]).to(self.device), torch.tensor([1.0]).to(self.device)

        self.fc1 = PyroModule[nn.Linear](self.num_features, model_spec['embed'])
        self.drop = pyro.sample('drop', dist.Categorical(probs=torch.tensor([0.5, 0.5]).expand([model_spec['embed'], self.num_features, 2]))).float()
        self.fc1.weight = PyroSample(dist.Normal(mu, 0.0000001+self.drop).expand([model_spec['embed'], self.num_features]).to_event(2))
        self.fc1.bias = PyroSample(dist.Normal(mu, sigma).expand([model_spec['embed']]).to_event(1))

        self.fc2 = PyroModule[nn.Linear](model_spec['embed'], 2)
        self.fc2.weight = PyroSample(dist.Normal(mu, sigma).expand([2, model_spec['embed']]).to_event(2))
        self.fc2.bias = PyroSample(dist.Normal(mu, sigma).expand([2]).to_event(1))

        self.model = torch.nn.Sequential(self.fc1, self.fc2)

        self.to(self.device)

    def forward(self, x=None, y=None, seq_len=1):
        if x is None:
            with pyro.plate("x_plate", seq_len):
                d_ = dist.Normal(torch.tensor([0.0]).to(self.device), torch.tensor([1.0]).to(self.device)).expand(
                    [self.num_features]).to_event(1)
                x = pyro.sample("x", d_)

        out = self.model(x)
        mu = out.squeeze()
        softmax = torch.nn.Softmax(dim=1)
        # sigma = pyro.sample("sigma", dist.Uniform(torch.tensor([0.0]).to(self.device), torch.tensor([1.0]).to(self.device)))
        with pyro.plate("data", out.shape[0]):
            # d_ = dist.Normal(mu, sigma)
            # obs = pyro.sample("obs", d_, obs=y)
            s = softmax(mu)
            obs = pyro.sample('obs', dist.Categorical(probs=s), obs=y).float()

        return x, obs

class BayesianModel(PyroModule):
    def __init__(self, model_spec, device='cuda'):
        super().__init__()

        self.device = device
        self.num_features = model_spec['num_features']

        mu, sigma = torch.tensor([0.0]).to(self.device), torch.tensor([1.0]).to(self.device)

        self.fc1 = PyroModule[nn.Linear](self.num_features, model_spec['embed'])
        self.fc1.weight = PyroSample(
            dist.Normal(mu, sigma).expand([model_spec['embed'], self.num_features]).to_event(2))
        self.fc1.bias = PyroSample(dist.Normal(mu, sigma).expand([model_spec['embed']]).to_event(1))

        self.fc2 = PyroModule[nn.Linear](model_spec['embed'], 2)
        self.fc2.weight = PyroSample(dist.Normal(mu, sigma).expand([2, model_spec['embed']]).to_event(2))
        self.fc2.bias = PyroSample(dist.Normal(mu, sigma).expand([2]).to_event(1))

        self.model = torch.nn.Sequential(self.fc1, self.fc2)

        self.to(self.device)

    def forward(self, x=None, y=None, seq_len=1):
        if x is None:
            with pyro.plate("x_plate", seq_len):
                d_ = dist.Normal(torch.tensor([0.0]).to(self.device), torch.tensor([1.0]).to(self.device)).expand(
                    [self.num_features]).to_event(1)
                x = pyro.sample("x", d_)

        out = self.model(x)
        mu = out.squeeze()
        softmax = torch.nn.Softmax(dim=1)
        # sigma = pyro.sample("sigma", dist.Uniform(torch.tensor([0.0]).to(self.device), torch.tensor([1.0]).to(self.device)))
        with pyro.plate("data", out.shape[0]):
            # d_ = dist.Normal(mu, sigma)
            # obs = pyro.sample("obs", d_, obs=y)
            s = softmax(mu)
            obs = pyro.sample('obs', dist.Categorical(probs=s), obs=y).float()

        return x, obs


def get_transformer_config(model_spec):
    return {'lr': 2.006434218345026e-05
        , 'epochs': 400
        , 'dropout': 0.0
        , 'emsize': 256
        , 'batch_size': 256
        , 'nlayers': 5
        , 'num_outputs': 1
        , 'num_features': model_spec['num_features']
        , 'steps_per_epoch': 100
        , 'nhead': 4
        , 'dropout': 0.0
        , 'seq_len': model_spec['seq_len']
        , 'nhid_factor': 2}


def get_model(model_generator, config, should_train=True, device='cuda'):
    epochs = 0 if not should_train else config['epochs']

    model = train(priors.pyro.DataLoader
                  , Losses.bce
                  , encoders.Linear
                  , emsize=config['emsize']
                  , nhead=config['nhead']
                  , y_encoder_generator=encoders.Linear
                  , pos_encoder_generator=None
                  , batch_size=config['batch_size']
                  , nlayers=config['nlayers']
                  , nhid=config['emsize'] * config['nhid_factor']
                  , epochs=epochs
                  , warmup_epochs=config['epochs'] // 4
                  , bptt=config['seq_len']
                  , gpu_device=device
                  , dropout=config['dropout']
                  , steps_per_epoch=config['steps_per_epoch']
                  , single_eval_pos_gen=get_weighted_single_eval_pos_sampler(100)
                  , extra_prior_kwargs_dict={
            'num_outputs': config['num_outputs']
            , 'num_features': config['num_features']
            , 'canonical_args': None
            , 'fuse_x_y': False
            , 'model': model_generator
        }
                  , lr=config['lr']
                  , verbose=True)

    return model



def plot_features(data, targets):
    fig2 = plt.figure(constrained_layout=True, figsize=(12, 12))
    spec2 = gridspec.GridSpec(ncols=data.shape[1], nrows=data.shape[1], figure=fig2)
    for d in range(0, data.shape[1]):
        for d2 in range(0, data.shape[1]):
            sub_ax = fig2.add_subplot(spec2[d, d2])
            sub_ax.scatter(data[:, d].detach().cpu().numpy(), data[:, d2].detach().cpu().numpy(),
                           c=targets[:].detach().cpu().numpy())


def evaluate_preds(preds, y_test):
    preds_hard = preds['obs'] > 0.5  # TODO: 0.5 or 0
    acc = (preds_hard == y_test).float().mean()
    means = preds_hard.float().mean(axis=0)

    # var = preds['obs'].var(axis=0)
    nll = nn.BCELoss()(means.float(), y_test.float())
    mse = Losses.mse(means, y_test).mean()

    return acc, nll, mse


def load_results(path, task='steps'):
    results_nll = []
    results_acc = []
    times = []
    samples_list = []

    files = glob.glob(f'/home/anon/prior-fitting/{path}_*.npy')
    for file in files:
        print(file)
        with open(file, 'rb') as f:
            if task == 'steps':
                nll, acc, elapsed = np.load(f, allow_pickle=True)
                samples_list += [file]
            else:
                samples, nll, acc, elapsed = np.load(f, allow_pickle=True)
                samples_list += [samples]
            times += [elapsed]
            results_nll += [nll]
            results_acc += [acc]
    results_acc = np.array(results_acc)
    results_nll = np.array(results_nll)
    times = np.array(times)
    files = np.array(files)
    samples = np.array(samples_list)
    means = np.array([compute_mean_and_conf_interval(results_nll[n, :])[0] for n in range(0, results_nll.shape[0])])
    conf = np.array([compute_mean_and_conf_interval(results_nll[n, :])[1] for n in range(0, results_nll.shape[0])])

    if task == 'steps':
        sorter = np.argsort(times, axis=0)
    else:
        sorter = np.argsort(samples, axis=0)

    results_nll, results_acc, times, files, samples, means, conf = results_nll[sorter], results_acc[sorter], times[sorter], files[sorter], samples[sorter], means[sorter], conf[sorter]

    return files, times, samples, means, conf

def plot_with_confidence_intervals(ax_or_pyplot, x, mean, confidence, **common_kwargs):
    ax_or_pyplot.plot(x,mean,**common_kwargs)
    if 'label' in common_kwargs:
        common_kwargs.pop('label')
    if 'marker' in common_kwargs:
        common_kwargs.pop('marker')
    ax_or_pyplot.fill_between(x, (mean-confidence), (mean+confidence), alpha=.1, **common_kwargs)


def compute_mean_and_conf_interval(accuracies, confidence=.95):
    accuracies = np.array(accuracies)
    n = len(accuracies)
    m, se = np.mean(accuracies), st.sem(accuracies)
    h = se * st.t.ppf((1 + confidence) / 2., n - 1)
    return m, h


def generate_toy_data(model, bptt, device='cpu'):
    n_samples = 100
    X_list, y_list = [], []
    torch.manual_seed(0)
    for _ in range(0, n_samples):
        X_sample, y_sample = model(seq_len=bptt)
        X_list += [X_sample]
        y_list += [y_sample]
    X = torch.stack(X_list, 0)
    y = torch.stack(y_list, 0)
    # y = (y > 0).float()

    return X.to(device), y.to(device)



def eval_svi(X, y, device, model_sampler, training_samples_n, num_train_steps, num_pred_samples, lr=1e-3, num_particles=1, svgd=False):
    X_test, y_test = X[:, training_samples_n:], y[:, training_samples_n:]
    X_train, y_train = X[:, 0:training_samples_n], y[:, 0:training_samples_n]

    nll_list = []
    acc_list = []
    for sample_id in tqdm(list(range(0, X_test.shape[0]))):
        model = model_sampler()
        guide = AutoDiagonalNormal(model).to(device)
        adam = pyro.optim.Adam({"lr": lr})
        svi = SVI(model, guide, adam, loss=Trace_ELBO(num_particles=num_particles))

        if svgd:
            kernel = RBFSteinKernel()
            svi = SVGD(model, kernel, adam, num_particles=50, max_plate_nesting=0)

        pyro.clear_param_store()

        X_test_sample, y_test_sample, X_train_sample, y_train_sample = X_test[sample_id], y_test[sample_id], X_train[
            sample_id], y_train[sample_id]

        acc, nll, mse = 0.0, 0.0, 0.0
        # bar = tqdm(list(range(num_train_steps)))
        bar = list(range(num_train_steps))
        for epoch in bar:
            loss = svi.step(X_train_sample, y_train_sample)
            # if epoch % 100 == 1:
            #    bar.set_postfix(loss=f'{loss / X_train_sample.shape[0]:.3f}', test_nll=f'{nll:.3f}', test_acc=f'{acc:.3f}')

        predictive = Predictive(model, guide=guide, num_samples=num_pred_samples)
        preds = predictive(X_test_sample)
        acc, nll, mse = evaluate_preds(preds, y_test_sample)
        nll_list += [nll.detach().cpu().numpy()]
        acc_list += [acc.detach().cpu().numpy()]

    return np.array(nll_list), np.array(acc_list)


def eval_mcmc(X, y, device, model_sampler, training_samples_n, warmup_steps, num_pred_samples):
    X_test, y_test = X[:, training_samples_n:].to(device), y[:, training_samples_n:].to(device)
    X_train, y_train = X[:, 0:training_samples_n].to(device), y[:, 0:training_samples_n].to(device)

    acc_list, nll_list = [], []
    for sample_id in tqdm(list(range(0, X_test.shape[0]))):
        X_test_sample, y_test_sample, X_train_sample, y_train_sample = X_test[sample_id], y_test[sample_id], X_train[
            sample_id], y_train[sample_id]

        model = model_sampler()
        mcmc = MCMC(NUTS(model), num_samples=num_pred_samples, num_chains=1, disable_progbar=True,
                    warmup_steps=warmup_steps, mp_context="fork")
        mcmc.run(X_train_sample, y_train_sample)
        preds = infer.mcmc.util.predictive(model, mcmc.get_samples(), X_test_sample, None)
        acc, nll, mse = evaluate_preds(preds, y_test_sample)
        nll_list += [nll.detach().cpu().numpy()]
        acc_list += [acc.detach().cpu().numpy()]

    return np.array(nll_list), np.array(acc_list)


def eval_transformer(X, y, device, model, training_samples_n):
    X_sample, y_sample = X.transpose(0, 1), y.transpose(0, 1).float()
    bs = 1
    samples = []
    for i in range(0, X_sample.shape[1] // bs):
        samples += [(X_sample[:, bs * i:bs * (i + 1)], y_sample[:, bs * i:bs * (i + 1)])]

    mean = X_sample[:training_samples_n].mean(0)
    std = X_sample[:training_samples_n].std(0) + .000001
    X_sample = (X_sample - mean) / std

    start = time.time()
    output = torch.cat(
        [model.to(device)((X_sample_chunk, y_sample_chunk), single_eval_pos=training_samples_n).squeeze(-1) for
         (X_sample_chunk, y_sample_chunk) in samples], 1)
    elapsed = time.time() - start

    output = output.detach().cpu()
    acc = ((torch.sigmoid(output) > 0.5) == y_sample[training_samples_n:].cpu().bool()).float().mean(axis=0)
    nll = nn.BCELoss(reduction='none')(torch.sigmoid(output.float()), y_sample[training_samples_n:].cpu().float()).mean(
        axis=0)
    return acc, nll, elapsed


def training_steps(method, X, y, model_spec, device='cpu', path_interfix='', overwrite=False):
    training_samples_n = 100
    for s in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
        path = f'/home/anon/prior-fitting/{path_interfix}/results_{method}_training_steps_{s}.npy'
        if (os.path.isfile(path)) and not overwrite:
            print(f'already done {s}')
            continue

        start = time.time()
        if method == 'svi':
            nll, acc = eval_svi(X, y, device, model_spec, training_samples_n, num_train_steps=s, num_pred_samples=s, svgd=False)
        elif method == 'svgd':
            nll, acc = eval_svi(X, y, device, model_spec, training_samples_n, num_train_steps=s, num_pred_samples=s, svgd=True)
        elif method == 'mcmc':
            nll, acc = eval_mcmc(X, y, device, model_spec, training_samples_n, warmup_steps=s, num_pred_samples=s)
        elapsed = time.time() - start

        print(s)
        print('NLL ', compute_mean_and_conf_interval(nll))
        print('ACC ', compute_mean_and_conf_interval(acc))
        print('TIME ', elapsed)

        with open(path, 'wb') as f:
            np.save(f, (np.array(nll), np.array(acc), elapsed))

        print(f'Saved results at {path}')


def training_samples(method, X, y, model_spec, evaluation_points, steps = None, device='cpu', path_interfix='', overwrite=False):
    num_pred_samples_mcmc = steps if steps else 512
    warmup_steps = steps if steps else 512

    num_pred_samples_svi = steps if steps else 1024
    num_train_steps = steps if steps else 1024

    num_pred_samples = num_pred_samples_svi if method == 'svi' else num_pred_samples_mcmc

    for training_samples_n in evaluation_points:
        path = f'/home/anon/prior-fitting/{path_interfix}/results_{method}_{num_pred_samples}_training_samples_{training_samples_n}.npy'
        if (os.path.isfile(path)) and not overwrite:
            print(f'already done {training_samples_n}')
            continue

        start = time.time()
        if method == 'svi':
            nll, acc = eval_svi(X, y, device, model_spec, training_samples_n, num_train_steps=num_train_steps, num_pred_samples=num_pred_samples)
        elif method == 'svgd':
            nll, acc = eval_svi(X, y, device, model_spec, training_samples_n, num_train_steps=num_train_steps, num_pred_samples=num_pred_samples, svgd=True)
        elif method == 'mcmc':
            nll, acc = eval_mcmc(X, y, device, model_spec, training_samples_n, warmup_steps=warmup_steps, num_pred_samples=num_pred_samples)
        elapsed = time.time() - start

        print('NLL ', compute_mean_and_conf_interval(nll))
        print('ACC ', compute_mean_and_conf_interval(acc))
        print('TIME ', elapsed)

        with open(path, 'wb') as f:
            np.save(f, (training_samples_n, np.array(nll), np.array(acc), elapsed))

### MAIN
def get_default_model_spec(size):
    bptt = 300

    if size == 'big':
        num_features = 8
        embed = 64
        nlayers = 2
    elif size == 'small':
        num_features = 3
        embed = 5
        nlayers = 2
    else:
        num_features = int(size.split("_")[0])
        embed = int(size.split("_")[1])
        nlayers = int(size.split("_")[2])

    return {'nlayers': nlayers, 'embed': embed, 'num_features': num_features, "seq_len": bptt}

def get_default_evaluation_points():
    return list(range(2, 100, 5))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--solver', default='svi', type=str)
    parser.add_argument('--task', default='steps', type=str)
    parser.add_argument('--model_size', default='small', type=str)

    args = parser.parse_args()

    model_spec = get_default_model_spec(args.model_size)
    evaluation_points = get_default_evaluation_points()
    device = 'cuda:0' if args.solver == 'svi' else 'cpu'

    torch.manual_seed(0)
    test_model = BayesianModel(model_spec, device=device)

    X, y = generate_toy_data(test_model, model_spec['seq_len'])
    model_sampler = lambda: BayesianModel(model_spec, device=device)

    if args.task == 'steps':
        training_steps(args.solver, X, y, model_sampler, device=device,
                       path_interfix=f'results/timing_{args.model_size}_model', svgd=args.svgd)
    elif args.task == 'samples':
        training_samples(args.solver, X, y, model_sampler, evaluation_points, device=device,
                       path_interfix=f'results/timing_{args.model_size}_model', svgd=args.svgd)



