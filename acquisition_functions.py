from botorch.acquisition import AcquisitionFunction
from torch import Tensor


class ExpectedImprovement(AcquisitionFunction):
    def forward(self, X: Tensor, best_f, maximize=True) -> Tensor: # X: evaluation_points x feature_dim
        assert len(X.shape) == 2

        model = self.get_submodule('model')

        y = model(X)

        full_range = model.full_range





