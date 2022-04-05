## The Official Code for the Paper "Transformers Can Do Bayesian Inference"

<img width="600" alt="Screen Shot 2022-02-15 at 19 32 22" src="https://user-images.githubusercontent.com/9828297/154126371-d54af7b8-a997-426d-838f-eeaf590c2276.png">

We train Transformers to do Bayesian Prediction on novel datasets for a large variety of priors. For more info read our [paper](https://arxiv.org/abs/2112.10510).
You can play with our model in an interactive [demo](https://huggingface.co/spaces/samuelinferences/transformers-can-do-bayesian-inference) with a GP prior and compare it to the ground truth GP posterior, as described in the paper's section 5.1.

For insights into experiments, please see our `notebooks` folder. From where most experiments, besides some baselines are started.

Training the transformers can be quickly done for all tasks considered, but we still provide models for the tabular tasks as convenience to be able solve new tabular tasks out-of-the-box.


__Getting Started__

This is a python project, we used Python 3.9 in development and recommend to use a `virtualenv` or `conda`.
To use our code, clone the project with

```
git clone git@github.com:automl/TransformersCanDoBayesianInference.git
```

install all dependencies with

```
pip install -r requirements.txt
```


__Reproducing the GP results__
You can have a look at [notebooks/SetupForGPFittingExperiments.ipynb](notebooks/SetupForGPFittingExperiments.ipynb). The hyper-paramters are chosen to reproduce figure 3 a). If you want to consider smaller datasets reduce `bptt` and the max number of training samples provided in `utils.get_weighted_single_eval_pos_sampler`.


__Training a model with a custom prior__

[notebooks/BayesianModels_And_Custom_Pyro_Modules.ipynb](notebooks/BayesianModels_And_Custom_Pyro_Modules.ipynb) provides a workflow to train and evaluate a PFN model with a custom prior. A prior is defined by providing a sampling procedure as a PyroModule. A prior template can be found in this notebook.

Below we show an overview of training a PFN for a custom prior. A full example can be found in BayesianModels_And_Custom_Pyro_Modules.ipynb.
```
class CustomModel(PyroModule):
    def __init__(self, device='cuda'):
        super().__init__()

        self.model = model_spec()

    def forward(self, seq_len=1):
        with pyro.plate("x_plate", seq_len):
            d_ = dist.Normal(torch.tensor([0.0]).to(self.device), torch.tensor([1.0]).to(self.device)).expand(
                [self.num_features]).to_event(1)
            x = pyro.sample("x", d_)

        out = self.model(x)
        
        return x, out
```

```
# Function which generates a model from the prior
model_sampler = lambda : BayesianModel(model_spec, device = device)
```

```
config = {'lr': 2.006434218345026e-05, 'epochs': 160}

transformer_model = get_model(model_sampler, config, should_train = True)
```

__Evaluating Tabular Models__

[notebooks/TabularEvalSimple.ipynb](notebooks/TabularEvalSimple.ipynb) provides a workflow to evaluate baselines and the transformer on the balanced subset of the AutoML Benchmark (filtered by Nans, number of features).
