"""State space models for CEM MPC using gaussian processes."""
from typing import Optional, Tuple, Dict, List, Any

import gpytorch
import torch
import torch.nn as nn
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.kernels import ScaleKernel, RBFKernel, LinearKernel, Kernel
from gpytorch.likelihoods import GaussianLikelihood
from torch import Tensor

from .ssm_cem import CemSSM
from ..ssm_pytorch import MultiOutputGP, utilities
from ..utils import get_device, assert_shape


class GpCemSSM(CemSSM):
    """A SSM using an exact GP from GPyTorch, for the CEM implementation of SafeMPC.

    Compared to GPyTorchSSM, this uses PyTorch tensors all the way through, rather than converting to Numpy. It also
    does not implement the linearization and differentiation functions which are only require for Casadi.
    """

    def __init__(self, conf, state_dimen: int, action_dimen: int, model: Optional[MultiOutputGP] = None):
        """Constructs a new instance.

        :param model: Set during unit testing to inject a model.
        """
        super().__init__(state_dimen, action_dimen)

        self._training_iterations = conf.exact_gp_training_iterations

        self._likelihood = GaussianLikelihood(batch_size=state_dimen, noise_constraint=gpytorch.constraints.GreaterThan(0.0),)
        if model is None:
            self._model = MultiOutputGP(train_x=None, train_y=None, likelihood=self._likelihood,
                                        kernel=self._create_kernel(conf, state_dimen, action_dimen),
                                        num_outputs=state_dimen)
        else:
            self._model = model
        self._model = self._model.to(get_device(conf))
        self._model.eval()

        self._last_training_losses = []

    def _create_kernel(self, conf, state_dimen: int, action_dimen: int) -> Kernel:
        if conf.exact_gp_kernel == 'rbf':
            kernel = RBFKernel(batch_size=state_dimen, ard_num_dims=state_dimen+action_dimen)
        elif conf.exact_gp_kernel == 'linear':
            kernel = LinearKernel(batch_size=state_dimen)
        elif conf.exact_gp_kernel == 'nn':
            kernel = NNFeatureKernel(batch_size=state_dimen, in_dimen=state_dimen + action_dimen,
                                     layer_sizes=conf.nn_kernel_layers)
        else:
            raise ValueError(f'Unknown kernel {conf.exact_gp_kernel}')

        # TODO: work out why we need ScaleKernel.
        return ScaleKernel(kernel, batch_size=state_dimen)

    def predict_with_jacobians(self, states: Tensor, actions: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        z = self._join_states_actions(states, actions)

        pred_mean, pred_var = self._predict(z)

        # TODO: avoid a second forward pass through the model.
        def mean_func(x: Tensor):
            return self._predict(x)[0]

        pred_mean_jac = utilities.compute_jacobian_fast(mean_func, z, num_outputs=self.num_states)

        N = states.size(0)
        assert_shape(pred_mean_jac, (N, self.num_states, self.num_states + self.num_actions))

        return pred_mean, pred_var, pred_mean_jac

    def predict_without_jacobians(self, states: Tensor, actions: Tensor) -> Tuple[Tensor, Tensor]:
        z = self._join_states_actions(states, actions)
        return self._predict(z)

    def _predict(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        pred_mean, pred_var = self.predict_raw(z)
        pred_mean = pred_mean.transpose(0, 1)
        pred_var = pred_var.transpose(0, 1)

        N = z.size(0)
        assert_shape(pred_mean, (N, self.num_states))
        assert_shape(pred_var, (N, self.num_states))

        return pred_mean, pred_var

    def predict_raw(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        N = z.size(0)
        assert_shape(z, (N, self.num_states + self.num_actions))
        pred = self._likelihood(self._model(z))
        return pred.mean, pred.variance

    def _update_model(self, x_train: Tensor, y_train: Tensor) -> None:
        # Hack because set_train_data() does not work if previous data was None.
        self._model.train_inputs = []
        self._model.train_targets = torch.zeros((0), device=x_train.device)

        self._model.set_train_data(x_train, y_train.transpose(0, 1), strict=False)

    def _train_model(self, x_train: Tensor, y_train: Tensor) -> None:
        self._model.train()
        self._likelihood.train()

        # self._model.parameters() includes likelihood parameters
        optimizer = torch.optim.Adam([{'params': self._model.parameters()}, ], lr=0.01)

        mll = ExactMarginalLogLikelihood(self._likelihood, self._model)

        print(f'Training GP on {self.x_train.size(0)} data points for {self._training_iterations} iterations...')
        losses = []
        for i in range(self._training_iterations):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = self._model(x_train)
            # Calc loss and backprop gradients
            loss = -mll(output, y_train.transpose(0, 1)).sum()
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
        print(f'Training complete. Final losses: {losses[-4]:.2f} {losses[-3]:.2f} {losses[-2]:.2f} {losses[-1]:.2f}')

        self._last_training_losses = losses

        self._model.eval()
        self._likelihood.eval()

    def collect_metrics(self) -> Dict[str, Any]:
        # Currently we don't have any metrics.
        return {'losses': self._last_training_losses}

    @property
    def parametric(self) -> bool:
        return False


class NNFeatureKernel(LinearKernel):
    """A dot product kernel, where the input points are transformed by a neural network.

    Implements k(x, x') = v phi(x)^T phi(x'), where phi is a fully connected network of some size.
    """

    def __init__(self, in_dimen: int, layer_sizes: List[int], **kwargs):
        super().__init__(**kwargs)
        self._in_dimen = in_dimen
        self._out_dimen = layer_sizes[-1]
        self._net = self._build_net(in_dimen, layer_sizes)

        for param_name, param_data in self._net.named_parameters():
            # GPyTorch param names can't contain ".".
            param_name = param_name.replace('.', '_')
            self.register_parameter(f'net_{param_name}', param_data)

    @staticmethod
    def _build_net(in_dimen: int, layer_sizes: List[int]) -> nn.Module:
        """Returns a fully connected network with a ReLU after ever layer, except PReLU after the final layer."""
        layers = []
        prev_size = in_dimen
        for i, layer_size in enumerate(layer_sizes):
            if i != 0:
                layers.append(nn.ReLU())
            layers.append(nn.Linear(prev_size, layer_size))
            prev_size = layer_size

        layers.append(nn.PReLU())

        return nn.Sequential(*layers)

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **kwargs):
        if last_dim_is_batch is not False:
            raise NotImplementedError

        x1_features = self._net(x1)
        x2_features = self._net(x2)

        # Normalize the features between 0 and 1.
        x1_scaled = x1_features - x1_features.min(2, keepdim=True)[0]
        x1_scaled = 2 * (x1_scaled / x1_features.max(2, keepdim=True)[0]) - 1
        x2_scaled = x2_features - x2_features.min(2, keepdim=True)[0]
        x2_scaled = 2 * (x2_scaled / x2_features.max(2, keepdim=True)[0]) - 1

        return super().forward(x1_scaled, x2_scaled, diag, last_dim_is_batch, **kwargs)
