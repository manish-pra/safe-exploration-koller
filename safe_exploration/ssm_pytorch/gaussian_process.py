"""Gaussian process utlilities for gpytorch."""
from typing import Optional

import gpytorch
# import hessian
import numpy as np
import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import Kernel
from gpytorch.likelihoods import Likelihood
from gpytorch.means import Mean
from gpytorch.mlls import MarginalLogLikelihood
from gpytorch.models import ExactGP
from torch import Tensor
from torch.nn import ModuleList

from safe_exploration.state_space_models import StateSpaceModel
from .utilities import compute_jacobian

__all__ = ['LinearMean', 'MultiOutputGP', 'GPyTorchSSM']


class LinearMean(gpytorch.means.Mean):
    """A linear mean function.
    If the matrix has more than one rows, the mean will be applied in batch-mode.
    Parameters
    ----------
    matrix : torch.tensor
        A 2d matrix. For each feature vector x in (d, 1) the output is `A @ x`.
    trainable : bool, optional
        Whether the mean matrix should be trainable as a parameter.
    prior : optional
        The gpytorch prior for the parameter. Ignored if trainable is False.
    """

    def __init__(self, matrix, trainable=False, prior=None):
        super().__init__()
        if trainable:
            self.register_parameter(name='matrix', parameter=torch.nn.Parameter(matrix))
            if prior is not None:
                self.register_prior('matrix_prior', prior, 'matrix')
        else:
            self.matrix = matrix

    @property
    def batch_size(self):
        return self.matrix.size(0)

    def forward(self, x):
        """Compute the linear product."""
        return torch.einsum('ij,ilj->il', self.matrix, x)


class WrappedNormal(object):
    """A wrapper around gpytorch.NormalDistribution that doesn't squeeze empty dims."""

    def __init__(self, normal):
        super().__init__()
        self.normal = normal

    def __getattr__(self, key):
        """Unsqueeze empty dimensions."""
        res = getattr(self.normal, key)
        batch_shape = self.normal.batch_shape
        if not batch_shape and key in ('mean', 'variance', 'covariance_matrix'):
            res = res.unsqueeze(0)

        return res


class ZeroMeanWithGrad(Mean):
    """A zero mean like gpytorch.means.ZeroMean, but maintains the requires_grad state of the input.

    This is because we often want to compute the jacobian of the output of the GP. This requires the output, and thus
    the mean, to have a gradient.
    """

    def forward(self, x: Tensor) -> Tensor:
        return torch.zeros((x.size(0), x.size(1)), dtype=x.dtype, device=x.device, requires_grad=x.requires_grad)


class MultiOutputGP(ExactGP):
    """A GP model that uses the gpytorch batch mode for multi-output predictions.

    The main difference to simple batch mode, is that the model assumes that all GPs
    use the same input data. Moreover, even for single-input data it outputs predictions
    together with a singular dimension for the batchsize.
    """

    def __init__(self, train_x: Optional[Tensor], train_y: Optional[Tensor], kernel: Kernel, likelihood: Likelihood,
                 num_outputs: int, mean: Mean = None):
        """Constructs a new instance.

        :param train_x: torch.tensor
        A (n x d) tensor with n data points of d dimensions each.
        :param train_y: torch.tensor
            A (n x o) tensor with n data points across o output dimensions.
        :param kernel: gpytorch.kernels.Kernel
            A kernel with appropriate batchsize.
        :param likelihood: gpytorch.likelihoods.Likelihood
            A GP likelihood with appropriate batchsize.
        :param mean: gpytorch.means.Mean, optional
            The mean function with appropriate batchsize. See `BatchMean`. Defaults to
            `gpytorch.means.ZeroMeanWithGrad()`.
        """
        self._assert_training_data_shape(train_x, train_y)
        super().__init__(train_x, train_y, likelihood)

        self._mean = mean if mean is not None else ZeroMeanWithGrad()
        self._kernel = kernel
        self._num_outputs = num_outputs

    @property
    def batch_size(self):
        """Return the number of outputs of the model."""
        return self._num_outputs

    def set_train_data(self, inputs=None, targets=None, strict=True):
        """Set the GP training data."""
        self._assert_training_data_shape(inputs, targets)
        super().set_train_data(inputs, targets, strict)

    @staticmethod
    def _assert_training_data_shape(train_x, train_y):
        if train_x is None or train_y is None:
            return

        assert (train_y.dim() == 1 and train_x.shape[0] == train_y.shape[0]) or train_x.shape[0] == train_y.shape[1], (
            f"We require x:[N x n] y:[N] or x:[N x n] y:[m x N]. We got x:{train_x.shape} y:{train_y.shape}")

    def loss(self, mml: MarginalLogLikelihood):
        """Return the negative log-likelihood of the model."""
        output = super().__call__(*self.train_inputs)
        return -mml(output, self.train_targets).sum()

    def forward(self, x):
        # x: [N x n], expand to [d x N x n] where n=input_dimen, d=output_dimen
        # We have a batch of GPs, one for each output dimension. We want each GP to get the same input, hence we expand.
        x = x.expand((self._num_outputs,) + x.size())
        return MultivariateNormal(self._mean(x), self._kernel(x))


class GPyTorchSSM(StateSpaceModel):
    """ A Gaussian process state space model based on GPytorch.

    We approximate the function x_{t+1} = f(x_t, u_t) with x in (1 x n) and u in (1 x m)
    based on noisy observations of f.

    """

    def __init__(self, num_states, num_actions, train_x, train_y, kernel, likelihood, mean=None):
        """ """

        # check compatability of the parameters required for super classes
        assert np.shape(train_x)[1] == num_states + num_actions, "Input needs to have dimensions N x(n + m)"
        assert np.shape(train_y)[0] == num_states, "Input needs to have dimensions N x n"

        self.pytorch_gp = MultiOutputGP(train_x, train_y, kernel, likelihood, mean)
        self.pytorch_gp.eval()

        super(GPyTorchSSM, self).__init__(num_states, num_actions, True, True)

    def _compute_hessian_mean(self, states, actions):
        """ Generate the hessian of the mean prediction

        Parameters
        ----------
        states : np.ndarray
            A (1 x n) array of states.
        actions : np.ndarray
            A (1 x m) array of actions.

        Returns
        -------
        hess_mean:

        """

        inp = torch.cat((torch.from_numpy(np.array(states, dtype=np.float32)),
                         torch.from_numpy(np.array(actions, dtype=np.float32))), dim=1)
        inp.requires_grad = True
        n_in = self.num_states + self.num_actions

        hess_mean = torch.empty(self.num_states, n_in, n_in)
        for i in range(self.num_states):  # unfortunately hessian only works for scalar outputs
            hess_mean[i, :, :] = hessian.hessian(self.pytorch_gp(inp).mean[i, 0], inp)

        return hess_mean.numpy()

    def _predict(self, states, actions, jacobians=False, full_cov=False):
        """Predict the next states and uncertainty.

        Parameters
        ----------
        states : torch.tensor
            A (N x n) tensor of states.
        actions : torch.tensor
            A (N x m) tensor of actions.
        jacobians : bool, optional
            If true, return two additional outputs corresponding to the jacobians.
        full_cov : bool, optional
            Whether to return the full covariance.

        Returns
        -------
        mean : torch.tensor
            A (N x n) mean prediction for the next states.
        variance : torch.tensor
            A (N x n) variance prediction for the next states. If full_cov is True,
            then instead returns the (n x N x N) covariance matrix for each independent
            output of the GP model.
        jacobian_mean : torch.tensor
            A (N x n x n + m) tensor with the jacobians for each datapoint on the axes.
        jacobian_variance : torch.tensor
            Only supported without the full_cov flag.
        """
        if full_cov:
            raise NotImplementedError("Not implemented right now.")
        inp = torch.cat((states, actions), dim=1)
        inp.requires_grad = True
        self.inp = inp

        pred = self.pytorch_gp(inp)
        pred_mean = pred.mean
        pred_var = pred.variance

        if jacobians:
            jac_mean = compute_jacobian(pred_mean, inp).squeeze()
            jac_var = compute_jacobian(pred_var, inp).squeeze()

            return pred_mean, pred_var, jac_mean, jac_var

        else:
            self._forward_cache = torch.cat((pred_mean, pred_var))

            return pred_mean, pred_var

    def predict(self, states, actions, jacobians=False, full_cov=False):
        """Predict the next states and uncertainty.

        Parameters
        ----------
        states : np.ndarray
            A (N x n) array of states.
        actions : np.ndarray
            A (N x m) array of actions.
        jacobians : bool, optional
            If true, return two additional outputs corresponding to the jacobians.
        full_cov : bool, optional
            Whether to return the full covariance.

        Returns
        -------
        mean : np.ndarray
            A (N x n) mean prediction for the next states.
        variance : np.ndarray
            A (N x n) variance prediction for the next states. If full_cov is True,
            then instead returns the (n x N x N) covariance matrix for each independent
            output of the GP model.
        jacobian_mean : np.ndarray
            A (N x n x n + m) array with the jacobians for each datapoint on the axes.
        jacobian_variance : np.ndarray
            Only supported without the full_cov flag.
        """

        out = self._predict(torch.from_numpy(np.array(states, dtype=np.float32)),
                            torch.from_numpy(np.array(actions, dtype=np.float32)), jacobians, full_cov)
        return tuple([var.detach().numpy() for var in out])

    def linearize_predict(self, states, actions, jacobians=False, full_cov=False):
        """Predict the next states and uncertainty.

        Parameters
        ----------
        states : np.ndarray
            A (N x n) array of states.
        actions : np.ndarray
            A (N x m) array of actions.
        jacobians : bool, optional
            If true, return two additional outputs corresponding to the jacobians of the predictive
            mean, the linearized predictive mean and variance.
        full_cov : bool, optional
            Whether to return the full covariance.

        Returns
        -------
        mean : np.ndarray
            A (N x n) mean prediction for the next states.
        variance : np.ndarray
            A (N x n) variance prediction for the next states. If full_cov is True,
            then instead returns the (n x N x N) covariance matrix for each independent
            output of the GP model.
        jacobian_mean : np.ndarray
            A (N x n x (n + m) array with the jacobians for each datapoint on the axes.
        jacobian_variance : np.ndarray
            Only supported without the full_cov flag.
        hessian_mean: np.ndarray
            A (N x n*(n+m) x (n+m)) Array with the derivatives of each entry in the jacobian for each input
        """
        N, n = np.shape(states)

        if jacobians and N > 1:
            raise NotImplementedError("""'linearize_predict' currently only allows for single
                                          inputs, i.e. (1 x n) arrays, when computing jacobians.""")

        out = self._predict(torch.from_numpy(np.array(states, dtype=np.float32)),
                            torch.from_numpy(np.array(actions, dtype=np.float32)), True, full_cov)

        jac_mean = out[2]
        self._linearize_forward_cache = torch.cat((out[0], out[1], jac_mean.view(-1, 1)))

        out = [var.detach().numpy() for var in out]
        if jacobians:
            hess_mean = self._compute_hessian_mean(states, actions)

            return out[0], out[1], out[2], out[3], hess_mean

        else:

            return out[0], out[1], out[2]

    def get_reverse(self, seed):
        """ """
        self._forward_cache.backward(torch.from_numpy(seed), retain_graph=True)

        inp_grad = self.inp.grad
        grad_state = inp_grad[0, :self.num_states]

        grad_action = inp_grad[0, self.num_states:]

        return grad_state.detach().numpy(), grad_action.detach().numpy()

    def get_linearize_reverse(self, seed):
        """ """
        self._linearize_forward_cache.backward(torch.from_numpy(seed), retain_graph=True)
        inp_grad = self.inp.grad
        grad_state = inp_grad[0, :self.num_states].detach().numpy()[:, None]
        grad_action = inp_grad[0, self.num_states:].detach().numpy()[:, None]

        return grad_state, grad_action

    def update_model(self, train_x, train_y, opt_hyp=False, replace_old=False):
        raise NotImplementedError
