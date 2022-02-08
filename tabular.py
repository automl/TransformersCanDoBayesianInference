from tqdm import tqdm
import time
import random
import os
import argparse
import itertools

from torch import nn

import priors
from train import train, Losses
import encoders
from datasets import *
from priors.utils import trunc_norm_sampler_f, gamma_sampler_f

import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.infer import SVI, Trace_ELBO, Predictive, MCMC, NUTS

from catboost import CatBoostClassifier, Pool

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn import neighbors, datasets
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import accuracy_score, roc_auc_score

import xgboost as xgb
import matplotlib.pyplot as plt

CV = 5
param_grid = {}
metric_used = roc_auc_score

def get_uniform_single_eval_pos_sampler(max_len):
    """
    Just sample any evaluation position with the same weight
    :return: Sampler that can be fed to `train()` as `single_eval_pos_gen`.
    """
    return lambda: random.choices(range(max_len))[0]


def get_mlp_prior_hyperparameters(config):
    sigma_sampler = gamma_sampler_f(config["prior_sigma_gamma_k"], config["prior_sigma_gamma_theta"])
    noise_std_sampler = gamma_sampler_f(config["prior_noise_std_gamma_k"], config["prior_noise_std_gamma_theta"])

    mlp_prior_hyperparameters = (list(config["prior_nlayers_sampler"].values())[0]
                                 , list(config["prior_emsize_sampler"].values())[0]
                                 , config["prior_activations"]
                                 , sigma_sampler
                                 , noise_std_sampler
                                 , list(config["prior_dropout_sampler"].values())[0]
                                 , True
                                 , list(config["prior_num_features_used_sampler"].values())[0]
                                 , list(config["prior_causes_sampler"].values())[0] if config['prior_is_causal'] else None
                                 , config["prior_is_causal"]
                                 , config["prior_pre_sample_causes"] if config['prior_is_causal'] else None
                                 , config["prior_pre_sample_weights"] if config['prior_is_causal'] else None
                                 , config["prior_y_is_effect"] if config['prior_is_causal'] else None
                                 , config["prior_order_y"]
                                 , config["prior_normalize_by_used_features"]
                                 , list(config["prior_categorical_feats"].values())[0] if config['prior_is_causal'] else None
                                 , 0.0
                                 )

    return mlp_prior_hyperparameters


def get_gp_mix_prior_hyperparameters(config):
    return {'lengthscale_concentration': config["prior_lengthscale_concentration"],
            'nu': config["prior_nu"],
            'outputscale_concentration': config["prior_outputscale_concentration"],
            'categorical_data': config["prior_y_minmax_norm"],
            'y_minmax_norm': config["prior_lengthscale_concentration"],
            'noise_concentration': config["prior_noise_concentration"],
            'noise_rate': config["prior_noise_rate"]}


def get_gp_prior_hyperparameters(config):


    return (config['prior_noise']
            , lambda : config['prior_outputscale']
            , lambda : config['prior_lengthscale']  # lengthscale, Höher mehr sep
            , True
            , list(config['prior_num_features_used_sampler'].values())[0]
            , config['prior_normalize_by_used_features']
            , config['prior_order_y'])


def get_meta_gp_prior_hyperparameters(config):
    lengthscale_sampler = trunc_norm_sampler_f(config["prior_lengthscale_mean"], config["prior_lengthscale_mean"] * config["prior_lengthscale_std_f"])
    outputscale_sampler = trunc_norm_sampler_f(config["prior_outputscale_mean"], config["prior_outputscale_mean"] * config["prior_outputscale_std_f"])

    return (config['prior_noise']
            , outputscale_sampler
            , lengthscale_sampler  # lengthscale, Höher mehr sep
            , True
            , list(config['prior_num_features_used_sampler'].values())[0]
            , config['prior_normalize_by_used_features']
            , config['prior_order_y'])



def get_model(config, device, eval_positions, should_train=True, verbose=False):
    extra_kwargs = {}
    if config['prior_type'] == 'mlp':
        prior_hyperparameters = get_mlp_prior_hyperparameters(config)
        model_proto = priors.mlp.DataLoader
        extra_kwargs['batch_size_per_gp_sample'] = 8
    elif config['prior_type'] == 'gp':
        prior_hyperparameters = get_gp_prior_hyperparameters(config)
        model_proto = priors.fast_gp.DataLoader
    elif config['prior_type'] == 'custom_gp_mix':
        prior_hyperparameters = get_meta_gp_prior_hyperparameters(config)
        model_proto = priors.fast_gp.DataLoader
    elif config['prior_type'] == 'gp_mix':
        prior_hyperparameters = get_gp_mix_prior_hyperparameters(config)
        model_proto = priors.fast_gp_mix.DataLoader
    else:
        raise Exception()

    epochs = 0 if not should_train else config['epochs']
    model = train(model_proto
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
                  , warmup_epochs=epochs // 4
                  , bptt=config['bptt']
                  , gpu_device=device
                  , dropout=config['dropout']
                  , steps_per_epoch=100
                  , single_eval_pos_gen=get_uniform_single_eval_pos_sampler(max(eval_positions) + 1)
                  , extra_prior_kwargs_dict={
            'num_features': config['num_features']
            # , 'canonical_args': None
            , 'fuse_x_y': False
            , 'hyperparameters': prior_hyperparameters
            , **extra_kwargs
        }
                  , lr=config['lr']
                  , verbose=verbose)

    return model


## General eval

def evaluate(datasets, model, method, bptt, eval_position_range, device, max_features=0, plot=False, extend_features=False, save=True, rescale_features=False, overwrite=False,
             max_samples=40, path_interfix=''):
    # eval_position_range: last entry is the one used to calculate metricuracy; up to index is used for training
    result = {'metric': 'auc'}

    metric_sum = 0
    for [name, X, y, categorical_feats] in datasets:
        result_ds = {}
        path = f'/home/anon/prior-fitting/results/tabular/{path_interfix}/results_{method}_{name}.npy'
        if (os.path.isfile(path)) and not overwrite:
            with open(path, 'rb') as f:
                result_ds = np.load(f, allow_pickle=True).tolist()
                if 'time' in result_ds:
                    result_ds[name+'_time'] = result_ds['time']
                    del result_ds['time']
                result.update(result_ds)
                mean_metric = float(result[name + '_mean_metric_at_' + str(eval_position_range[-1])])
                metric_sum += mean_metric
                print(f'Loaded saved result for {name} (mean metric {mean_metric})')
                continue

        print('Evaluating ' + str(name))
        rescale_features_factor = X.shape[1] / max_features if rescale_features and extend_features else 1.0
        if extend_features:
            X = torch.cat([X, torch.zeros((X.shape[0], max_features - X.shape[1]))], -1)

        start_time = time.time()
        ds_result = evaluate_dataset(X.to(device), y.to(device), categorical_feats, model, bptt, eval_position_range,
                               rescale_features=rescale_features_factor, max_samples=max_samples)
        elapsed = time.time() - start_time

        for i, r in enumerate(ds_result):
            metric, outputs, ys = r
            if save:
                result_ds[name + '_per_ds_metric_at_' + str(eval_position_range[i])] = metric
                result_ds[name + '_outputs_at_' + str(eval_position_range[i])] = outputs
                result_ds[name + '_ys_at_' + str(eval_position_range[i])] = ys

            result_ds[name + '_mean_metric_at_' + str(eval_position_range[i])] = metric_used(ys.detach().cpu().flatten(), outputs.flatten())
            result_ds[name + '_time'] = elapsed

        if save:
            with open(path, 'wb') as f:
                np.save(f, result_ds)

        result.update(result_ds)
        metric_sum += float(metric[-1].mean())

    for pos in eval_position_range:
        result[f'mean_metric_at_{pos}'] = np.array([result[d[0] + '_mean_metric_at_' + str(pos)] for d in datasets]).mean()

    result['mean_metric'] = np.array([result['mean_metric_at_' + str(pos)] for pos in eval_position_range]).mean()

    return result


def evaluate_dataset(X, y, categorical_feats, model, bptt, eval_position_range, plot=False, rescale_features=1.0,
                     max_samples=40):
    result = []
    for eval_position in eval_position_range:
        r = evaluate_position(X, y, categorical_feats, model, bptt, eval_position, rescale_features=rescale_features,
                              max_samples=max_samples)
        result.append(r)
        print('\t Eval position ' + str(eval_position) + ' done..')

    if plot:
        plt.plot(np.array(list(eval_position_range)), np.array([r.mean() for r in result]))

    return result


def evaluate_position(X, y, categorical_feats, model, bptt, eval_position, rescale_features=1.0, max_samples=40):
    # right now permutation style is to test performance on one before the last element
    # eval_position = bptt - eval_positions

    # TODO: Make sure that no bias exists
    # assert(eval_position % 2 == 0)

    eval_xs = []
    eval_ys = []
    num_evals = len(X) - bptt  # len(X)-bptt-(bptt-eval_position)+1

    # Generate permutations of evaluation data
    #     with torch.random.fork_rng():
    #         torch.random.manual_seed(13)
    #         ps = [torch.randperm(2*(bptt - eval_position)) for _ in range(num_evals)]

    for i in range(num_evals):
        # Select chunk of data with extra evaluation positions that can be discarded
        #         x_ = X[i:i+bptt+(bptt-eval_position)].clone()
        #         y_ = y[i:i+bptt+(bptt-eval_position)].clone()

        #         # Permutate evaluation positions
        #         perm_range = slice(eval_position,bptt+(bptt - eval_position))
        #         x_[perm_range] = x_[perm_range][ps[i]]
        #         y_[perm_range] = y_[perm_range][ps[i]]

        #         # Discard extra evaluation positions
        #         x_ = x_[0:bptt]
        #         y_ = y_[0:bptt]

        x_ = X[i:i + bptt].clone()
        y_ = y[i:i + bptt].clone()

        eval_xs.append(x_)
        eval_ys.append(y_)

    # eval data will be ordered in training range and
    #   will be a random subset of 2*eval_position data points in eval positions
    eval_xs = torch.stack(eval_xs, 1)
    eval_ys = torch.stack(eval_ys, 1)

    # Limit to N samples per dataset
    with torch.random.fork_rng():
        torch.random.manual_seed(13)
        sel = torch.randperm(eval_xs.shape[1])
        eval_xs = eval_xs[:, sel[0:max_samples], :]
        eval_ys = eval_ys[:, sel[0:max_samples]]
    #
    # if quantile_transform:
    #     for sample in range(0, eval_xs.shape[1]):
    #         quantile_transformer = preprocessing.QuantileTransformer(random_state=0, n_quantiles=eval_xs.shape[0])
    #         quantile_transformer.fit(eval_xs[:eval_position, sample].cpu())
    #         eval_xs[:, sample] = torch.tensor(quantile_transformer.transform(eval_xs[:, sample].cpu()))

    if isinstance(model, nn.Module):
        model.eval()
        outputs = np.zeros(shape=(len(list(range(eval_position, eval_xs.shape[0]))), eval_xs.shape[1]))
        for i, pos in enumerate(range(eval_position, eval_xs.shape[0])):
            eval_x = torch.cat([eval_xs[:eval_position], eval_xs[pos].unsqueeze(0)])
            eval_y = eval_ys[:eval_position]

            # Center data using training positions so that it matches priors
            mean = eval_x.mean(0)
            std = eval_x.std(0) + .000001
            eval_x = (eval_x - mean) / std
            eval_x = eval_x / rescale_features

            output = torch.sigmoid(model((eval_x, eval_y.float()), single_eval_pos=eval_position)).squeeze(-1)
            outputs[i, :] = output.detach().cpu().numpy()

        metric_per_t = np.array([metric_used(eval_ys[eval_position:, i].cpu(), outputs[:, i]) for i in range(eval_xs.shape[1])])
        return metric_per_t, outputs, eval_ys[eval_position:]
    else:
        metric_eval_pos, outputs = batch_pred(model, eval_xs, eval_ys, categorical_feats, start=eval_position)

        return metric_eval_pos, outputs, eval_ys[eval_position:]


def batch_pred(metric_function, eval_xs, eval_ys, categorical_feats, start=2):
    metrics = []
    outputs = []
    # for i in tqdm(list(range(start,len(eval_xs)))):
    eval_splits = list(zip(eval_xs.transpose(0, 1), eval_ys.transpose(0, 1)))
    for eval_x, eval_y in tqdm(eval_splits):  # eval x is One sample i.e. bptt x num_features
        mean = eval_x[:start].mean(0)
        std = eval_x[:start].std(0) + .000001
        eval_x = (eval_x - mean) / std

        metric, output = metric_function(eval_x[:start], eval_y[:start], eval_x[start:], eval_y[start:], categorical_feats)
        metrics += [metric]
        outputs += [output]
    #     metrics_per_t.append(metric_sum/eval_xs.shape[1])
    return np.array(metrics), np.array(outputs).T

param_grid['logistic'] = {'solver': ['saga'], 'penalty': ['l1', 'l2', 'none'], 'tol': [1e-2, 1e-4, 1e-10], 'max_iter': [500], 'fit_intercept': [True, False], 'C': [1e-5, 0.001, 0.01, 0.1, 1.0, 2.0]} # 'normalize': [False],
def logistic_metric(x, y, test_x, test_y, cat_features):
    import warnings
    def warn(*args, **kwargs):
        pass

    warnings.warn = warn

    x, y, test_x, test_y = x.cpu(), y.cpu(), test_x.cpu(), test_y.cpu()

    clf = LogisticRegression()

    # create a dictionary of all values we want to test for n_neighbors
    # use gridsearch to test all values for n_neighbors
    clf = GridSearchCV(clf, param_grid['logistic'], cv=min(CV, x.shape[0]//2))
    # fit model to data
    clf.fit(x, y.long())

    pred = clf.predict_proba(test_x)[:, 1]
    metric = metric_used(test_y.cpu().numpy(), pred)

    return metric, pred


## KNN
param_grid['knn'] = {'n_neighbors (max number of samples)': np.arange(1, 6)}
def knn_metric(x, y, test_x, test_y, cat_features):
    x, y, test_x, test_y = x.cpu(), y.cpu(), test_x.cpu(), test_y.cpu()

    clf = neighbors.KNeighborsClassifier()  # min(param['n_neighbors'],len(y)))
    param_grid_knn = {'n_neighbors': np.arange(1, min(6, len(y) - 1))}
    # create a dictionary of all values we want to test for n_neighbors
    # use gridsearch to test all values for n_neighbors
    clf = GridSearchCV(clf, param_grid_knn, cv=min(CV, x.shape[0]//2))
    # fit model to data
    clf.fit(x, y.long())

    # print(clf.best_params_)

    # clf.fit(x, y.long())
    pred = clf.predict_proba(test_x)[:, 1]

    metric = metric_used(test_y.cpu().numpy(), pred)

    return metric, pred


## Bayesian NN
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


class BayesianNNClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, num_features, n_layers, embed, lr, device):
        self.num_pred_samples = 400
        self.num_steps = 400
        self.embed = embed
        self.n_layers = n_layers
        self.lr = lr
        self.num_features = num_features
        self.device = device

    def fit(self, X, y):
        model_spec = {'nlayers': 2, 'embed': self.embed, 'num_features': self.num_features}

        self.model = BayesianModel(model_spec, device=self.device)
        self.guide = AutoDiagonalNormal(self.model).to(self.device)
        self.adam = pyro.optim.Adam({"lr": self.lr})
        self.svi = SVI(self.model, self.guide, self.adam, loss=Trace_ELBO())

        pyro.clear_param_store()

        X = X.to(self.device)
        y = y.to(self.device)

        for epoch in tqdm(range(0, self.num_steps)):
            loss = self.svi.step(X, y)

        # Return the classifier
        return self

    def predict(self, X):
        X = X.to(self.device)
        predictive = Predictive(self.model, guide=self.guide, num_samples=self.num_pred_samples)
        preds = predictive(X)['obs']
        preds_means = preds.float().mean(axis=0).detach().cpu()
        preds_hard = preds_means > 0.5

        return preds_hard.long()

    def predict_proba(self, X):
        X = X.to(self.device)
        predictive = Predictive(self.model, guide=self.guide, num_samples=self.num_pred_samples)
        preds = predictive(X)['obs']
        preds_means = preds.float().mean(axis=0).detach().cpu()

        return preds_means

    def score(self, X, y):
        return super().score(X, y)

param_grid['bayes'] = {'embed': [5, 10, 30, 64], 'lr': [1e-3, 1e-4], 'num_training_steps': [400], 'num_samples_for_prediction': [400]}
def bayes_net_metric(x, y, test_x, test_y, cat_features):
    device = x.device

    clf = BayesianNNClassifier(x.shape[1], 2, 1, 1e-3, device)
    # create a dictionary of all values we want to test for n_neighbors
    # use gridsearch to test all values for n_neighbors
    clf = GridSearchCV(clf, param_grid['bayes'], cv=5)
    # fit model to data
    clf.fit(x.cpu(), y.long().cpu())

    pred = clf.predict_proba(test_x)
    metric = metric_used(test_y.cpu().numpy(), pred.cpu().numpy())

    return metric, pred

## GP
param_grid['gp'] = {'params_y_scale': [0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
    'params_length_scale': [0.1, 0.5, 1.0, 2.0]}
def gp_metric(x, y, test_x, test_y, cat_features):
    import warnings
    def warn(*args, **kwargs):
        pass
    warnings.warn = warn

    x, y, test_x, test_y = x.cpu(), y.cpu(), test_x.cpu(), test_y.cpu()

    clf = GaussianProcessClassifier()
    # create a dictionary of all values we want to test for n_neighbors
    params_y_scale = [0.05, 0.1, 0.5, 1.0, 5.0, 10.0]# 0.000001, 0.00001,
    params_length_scale = [0.1, 0.5, 1.0, 2.0] # 0.01,
    param_grid = {'kernel': [y * RBF(l) for (y, l) in list(itertools.product(params_y_scale, params_length_scale))]}
    # use gridsearch to test all values for n_neighbors
    clf = GridSearchCV(clf, param_grid, cv=min(CV, x.shape[0]//2))
    # fit model to data
    clf.fit(x, y.long())
    pred = clf.predict_proba(test_x)[:, 1]
    metric = metric_used(test_y.cpu().numpy(), pred)

    return metric, pred


## Tabnet
# https://github.com/dreamquark-ai/tabnet
# param_grid['tabnet'] = {'n_d': [2, 4], 'n_steps': [2,4,6], 'gamma': [1.3], 'optimizer_params': [{'lr': 2e-2}, {'lr': 2e-1}]}
# #param_grid['tabnet'] = {'n_d': [2], 'n_steps': [2], 'optimizer_params': [{'lr': 2e-2}, {'lr': 2e-1}]}
# def tabnet_metric(x, y, test_x, test_y, cat_features):
#     x, y, test_x, test_y = x.cpu().numpy(), y.cpu().numpy(), test_x.cpu().numpy(), test_y.cpu().numpy()
#
#     mean_metrics = []
#     mean_best_epochs = []
#
#     for params in list(ParameterGrid(param_grid['tabnet'])):
#         kf = KFold(n_splits=min(5, x.shape[0]//2), random_state=None, shuffle=False)
#         metrics = []
#         best_epochs = []
#         for train_index, test_index in kf.split(x):
#             X_train, X_valid, y_train, y_valid = x[train_index], x[test_index], y[train_index], y[test_index]
#
#             clf = TabNetClassifier(verbose=True, cat_idxs=cat_features, n_a=params['n_d'], **params)
#
#             clf.fit(
#                 X_train, y_train,
#                 #eval_set=[(X_valid, y_valid)], patience=15
#             )
#
#             metric = metric_used(test_y.cpu().numpy(), clf.predict(X_valid))
#             metrics += [metric]
#             #best_epochs += [clf.best_epoch]
#         mean_metrics += [np.array(metrics).mean()]
#         #mean_best_epochs += [np.array(best_epochs).mean().astype(int)]
#
#     mean_metrics = np.array(mean_metrics)
#     #mean_best_epochs = np.array(mean_best_epochs)
#     params_used = np.array(list(ParameterGrid(param_grid['tabnet'])))
#
#     best_idx = np.argmax(mean_metrics)
#     #print(params_used[best_idx])
#     clf = TabNetClassifier(cat_idxs=cat_features, **params_used[best_idx])
#
#     clf.fit(
#         x, y#, max_epochs=mean_best_epochs[best_idx]
#     )
#
#     pred = 1 - clf.predict_proba(test_x)[:,0]
#     metric = metric_used(test_y, pred)
#
#     #print(metric, clf.predict(test_x), pred)
#
#     return metric, pred


# Catboost
param_grid['catboost'] = {'learning_rate': [0.1, 0.5, 1.0],
            'depth': [2, 4, 7],
            'l2_leaf_reg': [0.0, 0.5, 1],
            'iterations': [10, 40, 70]}
def catboost_metric(x, y, test_x, test_y, categorical_feats):
    import warnings
    def warn(*args, **kwargs):
        pass

    warnings.warn = warn

    x, y, test_x, test_y = x.numpy(), y.numpy(), test_x.numpy(), test_y.numpy()

    def make_pd_from_np(x):
        data = pd.DataFrame(x)
        for c in categorical_feats:
            data.iloc[:, c] = data.iloc[:, c].astype('int')
        return data

    x = make_pd_from_np(x)
    test_x = make_pd_from_np(test_x)

    model = CatBoostClassifier(iterations=2,
                               depth=2,
                               learning_rate=1,
                               loss_function='Logloss',
                               logging_level='Silent')

    grid_search_result = model.grid_search(param_grid['catboost'],
                                           X=x,
                                           y=y,
                                           cv=5,
                                           plot=False,
                                           verbose=False)  # randomized_search with n_iter

    # model.fit(x, y)
    pred = model.predict_proba(test_x)[:, 1]
    metric = metric_used(test_y, pred)

    return metric, pred


# XGBoost
param_grid['xgb'] = {
        'min_child_weight': [0.5, 1.0],
        'learning_rate': [0.02, 0.2],
        #'gamma': [0.1, 0.2, 0.5, 1, 2],
        'subsample': [0.5, 0.8],
        'max_depth': [1, 2],
        'colsample_bytree': [0.8], #0.5,
        'eval_metric': ['logloss'],
        'n_estimators': [100]
    }
def xgb_metric(x, y, test_x, test_y, cat_features):
    x, y, test_x, test_y = x.numpy(), y.numpy().astype(int), test_x.numpy(), test_y.numpy().astype(int)

    clf = xgb.XGBClassifier(use_label_encoder=False)

    # {'num_round': [2,5,10,20], 'max_depth': [1, 2,4,6,8], 'eta': [.1, .01, .001], 'eval_metric': 'logloss'}
    # use gridsearch to test all values for n_neighbors
    clf = GridSearchCV(clf, param_grid['xgb'], cv=5, n_jobs=4, verbose=2)
    # fit model to data
    clf.fit(x, y.astype(int))

    print(clf.best_params_)

    # clf.fit(x, y.long())
    pred = clf.predict_proba(test_x)[:, 1]
    metrics = ((pred > 0.5) == test_y).astype(float).mean()
    return metrics, pred

def get_default_spec(test_datasets, valid_datasets):
    bptt = 100
    eval_positions = [30] #list(range(6, 42, 2))  # list(range(10, bptt-10, 20)) + [bptt-10]
    max_features = max([X.shape[1] for (_, X, _, _) in test_datasets] + [X.shape[1] for (_, X, _, _) in valid_datasets])
    max_samples = 20

    return bptt, eval_positions, max_features, max_samples

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', default='knn', type=str)
    parser.add_argument('--did', default=-1, type=int)
    parser.add_argument('--overwrite', default=False, type=bool)
    args = parser.parse_args()

    test_datasets, _ = load_openml_list(test_dids_classification)
    valid_datasets, _ = load_openml_list(valid_dids_classification)

    selector = 'test'
    ds = valid_datasets if selector == 'valid' else test_datasets
    if args.did > -1:
        ds = ds[args.did:args.did+1]

    bptt, eval_positions, max_features, max_samples = get_default_spec(test_datasets, valid_datasets)

    if args.method == 'bayes':
        clf = bayes_net_metric
        device = 'cpu'
    elif args.method == 'gp':
        clf = gp_metric
        device = 'cpu'
    elif args.method == 'knn':
        clf = knn_metric
        device = 'cpu'
    elif args.method == 'catboost':
        clf = catboost_metric
        device = 'cpu'
    elif args.method == 'xgb':
        # Uses lots of cpu so difficult to time
        clf = xgb_metric
        device = 'cpu'
    elif args.method == 'logistic':
        clf = logistic_metric
        device = 'cpu'
    else:
        clf = None
        device = 'cpu'

    start_time = time.time()
    result = evaluate(ds, clf, args.method, bptt, eval_positions, device=device, max_samples=max_samples, overwrite=args.overwrite, save=True)
    result['time_spent'] = time.time() - start_time

    with open(f'/home/anon/prior-fitting/results/tabular/results_{selector}_{args.method}.npy', 'wb') as f:
        np.save(f, result)
