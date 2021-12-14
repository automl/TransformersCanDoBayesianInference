from . import fast_gp, fast_gp_mix
from .utils import get_batch_to_dataloader

def regression_prior_to_binary(get_batch_function):

    def binarized_get_batch_function(*args, assert_on=False, **kwargs):
        x, y, target_y = get_batch_function(*args, **kwargs)
        if assert_on:
            assert y is target_y, "y == target_y is assumed by this function"
        y = y.sigmoid().bernoulli()
        return x, y, y

    return binarized_get_batch_function


Binarized_fast_gp_dataloader = get_batch_to_dataloader(regression_prior_to_binary(fast_gp.get_batch))
Binarized_fast_gp_dataloader.num_outputs = 1


Binarized_fast_gp_mix_dataloader = get_batch_to_dataloader(regression_prior_to_binary(fast_gp_mix.get_batch))
Binarized_fast_gp_mix_dataloader.num_outputs = 1
