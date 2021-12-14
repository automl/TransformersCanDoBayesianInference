import random

import torch
from torch import nn

import numpy as np

from utils import default_device
from .utils import get_batch_to_dataloader
from .utils import order_by_y, normalize_data, normalize_by_used_features_f, Binarize
from .utils import trunc_norm_sampler_f, beta_sampler_f, gamma_sampler_f, uniform_sampler_f, zipf_sampler_f, scaled_beta_sampler_f, uniform_int_sampler_f


def canonical_pre_processing(x, canonical_args):
    assert x.shape[2] == len(canonical_args)
    ranges = [torch.arange(num_classes).float() if num_classes is not None else None for num_classes in canonical_args]
    for feature_dim, rang in enumerate(ranges):
        if rang is not None:
            x[:, :, feature_dim] = (x[:, :, feature_dim] - rang.mean()) / rang.std()
    return x


DEFAULT_NUM_LAYERS = 2
DEFAULT_HIDDEN_DIM = 100
DEFAULT_ACTIVATION_MODULE = torch.nn.ReLU
DEFAULT_INIT_STD = .1
DEFAULT_HIDDEN_NOISE_STD = .1
DEFAULT_FIXED_DROPOUT = 0.
DEFAULT_IS_BINARY_CLASSIFICATION = False


class GaussianNoise(nn.Module):
    def __init__(self, std):
        super().__init__()
        self.std = std

    def forward(self, x):
        return x + torch.normal(torch.zeros_like(x), self.std)


def causes_sampler_f(num_causes_sampler):
    num_causes = num_causes_sampler()
    means = np.random.normal(0, 1, (num_causes))
    std = np.abs(np.random.normal(0, 1, (num_causes)) * means)
    return means, std

def categorical_features_sampler(max_features):
    features = []
    ordinal = []
    num_categorical_features_sampler = scaled_beta_sampler_f(0.5, .8, max_features, 0)
    is_ordinal_sampler = lambda : random.choice([True, False])
    classes_per_feature_sampler = scaled_beta_sampler_f(0.1, 2.0, 10, 1)
    classes_per_feature_sampler_ordinal = scaled_beta_sampler_f(0.1, 2.0, 200, 1)
    for i in range(0, num_categorical_features_sampler()):
        ordinal_s = is_ordinal_sampler()
        ordinal.append(ordinal_s)
        classes = classes_per_feature_sampler_ordinal() if ordinal_s else classes_per_feature_sampler()
        features.append(np.random.rand(classes))
    return features, ordinal


def get_batch(batch_size, seq_len, num_features, device=default_device, hyperparameters=(DEFAULT_NUM_LAYERS, DEFAULT_HIDDEN_DIM, DEFAULT_ACTIVATION_MODULE, DEFAULT_INIT_STD, DEFAULT_HIDDEN_NOISE_STD, DEFAULT_FIXED_DROPOUT, DEFAULT_IS_BINARY_CLASSIFICATION),
              batch_size_per_gp_sample=None, num_outputs=1, canonical_args=None, sampling='normal'):
    assert num_outputs == 1
    num_layers_sampler, hidden_dim_sampler, activation_module, init_std_sampler, noise_std_sampler, dropout_prob_sampler, is_binary_classification, num_features_used_sampler, causes_sampler, is_causal, pre_sample_causes, pre_sample_weights, y_is_effect, order_y, normalize_by_used_features, categorical_features_sampler, nan_prob = hyperparameters

    # if is_binary_classification:
    #     sample_batch_size = 100*batch_size
    # else:
    sample_batch_size = batch_size

    # if canonical_args is not None:
    #     assert len(canonical_args) == num_causes
    #     # should be list of [None, 2, 4] meaning scalar parameter, 2 classes, 4 classes
    #
    #     for feature_idx, num_classes in enumerate(canonical_args):
    #         if num_classes is not None:
    #             causes[:,:,feature_idx] = torch.randint(num_classes, (seq_len, sample_batch_size))
    #
    #     causes = canonical_pre_processing(causes, canonical_args)

    batch_size_per_gp_sample = batch_size_per_gp_sample or sample_batch_size // 8
    assert sample_batch_size % batch_size_per_gp_sample == 0, 'Please choose a batch_size divisible by batch_size_per_gp_sample.'
    num_models = sample_batch_size // batch_size_per_gp_sample
    # standard kaiming uniform init currently...

    def get_model():
        class MLP(torch.nn.Module):
            def __init__(self):
                super(MLP, self).__init__()

                self.dropout_prob = dropout_prob_sampler()
                self.noise_std = noise_std_sampler()
                self.init_std = init_std_sampler()
                self.num_features_used = num_features_used_sampler()
                self.categorical_features, self.categorical_features_is_ordinal = categorical_features_sampler(self.num_features_used)
                if is_causal:
                    self.causes = causes_sampler() if is_causal else self.num_features_used
                    self.causes = (torch.tensor(self.causes[0], device=device).unsqueeze(0).unsqueeze(0).tile((seq_len,1,1)), torch.tensor(self.causes[1], device=device).unsqueeze(0).unsqueeze(0).tile((seq_len,1,1)))
                    self.num_causes = self.causes[0].shape[2]
                else:
                    self.num_causes = self.num_features_used
                self.num_layers = num_layers_sampler()
                self.hidden_dim = hidden_dim_sampler()

                if is_causal:
                    self.hidden_dim = max(self.hidden_dim, 2 * self.num_features_used+1)

                #print('cat', self.categorical_features, self.categorical_features_is_ordinal, self.num_features_used)

                assert(self.num_layers > 2)

                self.layers = [nn.Linear(self.num_causes, self.hidden_dim, device=device)]
                self.layers += [module for layer_idx in range(self.num_layers-1) for module in [
                        nn.Sequential(*[
                            activation_module()
                            , nn.Linear(self.hidden_dim, num_outputs if layer_idx == self.num_layers - 2 else self.hidden_dim, device=device)
                            , GaussianNoise(torch.abs(torch.normal(torch.zeros((num_outputs if layer_idx == self.num_layers - 2 else self.hidden_dim),device=device), self.noise_std))) if pre_sample_weights else GaussianNoise(self.noise_std)
                        ])
                    ]]
                self.layers = nn.Sequential(*self.layers)

                self.binarizer = Binarize() if is_binary_classification else lambda x : x

                # Initialize Model parameters
                for i, p in enumerate(self.layers.parameters()):
                    dropout_prob = self.dropout_prob if i > 0 else 0.0
                    nn.init.normal_(p, std=self.init_std / (1. - dropout_prob))
                    with torch.no_grad():
                        p *= torch.bernoulli(torch.zeros_like(p) + 1. - dropout_prob)

            def forward(self):
                if sampling == 'normal':
                    if is_causal and pre_sample_causes:
                        causes = torch.normal(self.causes[0], self.causes[1].abs()).float()
                    else:
                        causes = torch.normal(0., 1., (seq_len, 1, self.num_causes), device=device).float()
                elif sampling == 'uniform':
                    causes = torch.rand((seq_len, 1, self.num_causes), device=device)
                else:
                    raise ValueError(f'Sampling is set to invalid setting: {sampling}.')

                outputs = [causes]
                for layer in self.layers:
                    outputs.append(layer(outputs[-1]))
                outputs = outputs[2:]

                if is_causal:
                    outputs_flat = torch.cat(outputs, -1)
                    random_perm = torch.randperm(outputs_flat.shape[-1]-1, device=device)
                    random_idx_y = [-1] if y_is_effect else random_perm[0:num_outputs]
                    y = outputs_flat[:, :, random_idx_y]

                    random_idx = random_perm[num_outputs:num_outputs + self.num_features_used]
                    x = outputs_flat[:, :, random_idx]
                else:
                    y = outputs[-1][:, :, :]
                    x = causes

                if len(self.categorical_features) > 0:
                    random_perm = torch.randperm(x.shape[-1], device=device)
                    for i, (categorical_feature, is_ordinal) in enumerate(zip(self.categorical_features, self.categorical_features_is_ordinal)):
                        idx = random_perm[i]
                        temp = normalize_data(x[:, :, idx])
                        if is_ordinal:
                            x[:, :, idx] = (temp > (torch.tensor(categorical_feature, device=device, dtype=torch.float32).unsqueeze(-1).unsqueeze(-1) - 0.5)).sum(axis=0)
                        else:
                            x[:, :, idx] = (temp > (torch.tensor(categorical_feature, device=device,
                                                                dtype=torch.float32).unsqueeze(-1).unsqueeze(-1) - 0.5)).sum(
                                axis=0) * (127 * len(categorical_feature) + 1) % len(categorical_feature)


                # if nan_prob > 0:
                #     nan_value = random.choice([-999,-1,0, -10])
                #     x[torch.rand(x.shape, device=device) > (1-nan_prob)] = nan_value

                x, y = normalize_data(x), normalize_data(y)

                # Binarize output if enabled
                y = self.binarizer(y)

                if normalize_by_used_features:
                    x = normalize_by_used_features_f(x, self.num_features_used, num_features)

                if is_binary_classification and order_y:
                    x, y = order_by_y(x,y)

                # Append empty features if enabled
                x = torch.cat([x, torch.zeros((x.shape[0], x.shape[1], num_features - self.num_features_used), device=device)], -1)

                return x, y

        return MLP()

    models = [get_model() for _ in range(num_models)]

    sample = sum([[model() for _ in range(0,batch_size_per_gp_sample)] for model in models],[])

    x, y = zip(*sample)
    y = torch.cat(y, 1).squeeze(-1).detach()
    x = torch.cat(x, 1).detach()

    return x, y, y


DataLoader = get_batch_to_dataloader(get_batch)
DataLoader.num_outputs = 1

