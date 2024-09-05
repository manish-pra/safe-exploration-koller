# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 17:16:45 2017

@author: tkoller
"""
import abc
import itertools
import math
import warnings
from abc import abstractmethod
from typing import Tuple, Optional, Dict, Any, List

import numpy as np
import torch
from casadi import reshape as cas_reshape
from matplotlib.axes import Axes
from numpy import ndarray
from numpy.matlib import repmat
# from polytope import Polytope
from scipy.integrate import ode, odeint
from scipy.signal import cont2discrete
from scipy.spatial import ConvexHull
from torch import Tensor

from ..utils import unavailable, assert_shape
from ..visualization.utils_visualization import plot_ellipsoid_2D

try:
    import matplotlib.pyplot as plt

    _has_matplotlib = True
except:
    _has_matplotlib = False

try:
    import pygame

    _has_pygame = True
except:
    _has_pygame = False


class Environment(metaclass=abc.ABCMeta):
    """ Base class for environments


    Attributes
        ----------
        name: str
            The name of the environment
        n_s: int
            Number of state dimensions
        n_u: int
            Number of control dimensions
        dt: float
            time difference between simulation steps (i.e. 1/(simulation frequency))
        is_initialized: Bool
            True, if the environment is already initialized (e.g. via reset())
        iteration: int
            Number of timesteps for which the simulation is running (can be resetted
            via reset())
        init_std: n_s x 0 np.array[float]
        init_m
        u_min
        u_max
        plant_noise
        target
        verbosity
    """

    def __init__(self, name: str, n_s: int, n_u: int, dt: float, init_m: ndarray, init_std: ndarray,
                 plant_noise: ndarray, u_min: ndarray, u_max: ndarray, target: ndarray, verbosity: int = 0,
                 p_origin: ndarray = None):
        self.name = name
        self.n_s = n_s
        self.n_u = n_u
        self.dt = dt
        self.is_initialized = False
        self.iteration = 0
        self.init_std = init_std
        self.init_m = init_m
        self.u_min = u_min
        self.u_max = u_max
        self.plant_noise = plant_noise
        self.target = target
        self.verbosity = verbosity
        self._render_initialized = False
        self.delay = 20.0  # fps
        self.p_origin = p_origin if p_origin is not None else np.zeros((n_s,))
        self.current_episode_trajectory = []

    def reset(self, mean=None, std=None):
        """ Reset the system and sample a new start state."""
        self.is_initialized = True
        self.iteration = 0
        self.current_state = self._sample_start_state(mean=mean, std=std)
        self.current_episode_trajectory = [self.current_state]

        self._reset()

        return self.state_to_obs(self.current_state)

    @property
    @abstractmethod
    def l_mu(self) -> ndarray:
        """The Lipschitz constants for the gradients of the predictive mean.

        TODO: should be somewhere else.
        """
        pass

    @property
    @abstractmethod
    def l_sigm(self) -> ndarray:
        """The Lipschitz constants for the predictive variance.

        TODO: should be somewhere else.
        """
        pass

    @property
    def u_min_norm(self) -> ndarray:
        return self.inv_norm[1] * self.u_min

    @property
    def u_max_norm(self) -> ndarray:
        return self.inv_norm[1] * self.u_max

    @abstractmethod
    def _reset(self):
        pass

    @abstractmethod
    def _dynamics(self, t, state, action):
        """ Evaluate the system dynamics """
        pass

    @abstractmethod
    def state_to_obs(self, state=None, add_noise=False):
        """ Transform the dynamics state to the state to be observed """
        pass

    @abstractmethod
    def random_action(self) -> ndarray:
        """ Returns a uniform randon action. """
        pass

    def objective_cost_function(self, ps: Tensor) -> Optional[Tensor]:
        """Objective function which CEM MPC should optimise the trajectory with.

        :param ps: [Nxn_s] batch of centers of ellipsoids for one step in a trajectory
        :returns: [Nx0] cost of each of the centers, or None if the environment does not implement the cost and the mpc
        should use a default cost
        """
        return None

    @abstractmethod
    def plot_ellipsoid_trajectory(self, p, q, vis_safety_bounds=True):
        """ Visualize the reachability ellipsoid"""
        pass

    def plot_current_trajectory(self, axes: Axes) -> bool:
        """Plots the current episode trajectory to the given axes, if implemented by the subclass.

        :returns: True if something was plotted, otherwise False
        """
        return False

    def plot_states(self, axes: Axes, states: List[ndarray], includes_initial_samples: bool) -> bool:
        """Plots the given list of states on the constraints.

        Each element in the list is the states visited in a particular episode, where the first may be the initial
        samples (if enabled).

        :param includes_initial_samples: whether the first element of the list is the initial samples
        :returns: True if something was plotted, otherwise False
        """
        return False

    @abstractmethod
    def _jac_dynamics(self):
        """ The jacobian of the dynamics """
        pass

    @abstractmethod
    def _check_current_state(self, state: Optional[ndarray] = None) -> Tuple[bool, int]:
        """Check the given or current state to see if the episode should end e.g. because of a constraint violation.

        :param state: Some (unnormalized) state or the current state, if None then the current state is used
        :returns:
            done: True if the episode should end, otherwise false
            result code: status code as defined by the subclass
        """
        pass

    @abstractmethod
    def get_safety_constraints(self, normalize=True):
        """ Return the safety constraints

        Parameters
        ----------
        normalize: boolean, optional
            If TRUE: Returns normalized constraints

        Returns
        -------
        h_mat_safe: m_safe x n_x np.ndarray[float]
            The constraint matrix of the safety polytope
        h_safe: m_safe x 1 np.ndarray[float]
            The constraint vector of the safety polytope
        h_mat_obs: m_obs x n_x np.ndarray[float]
            The constraint matrix of the state constraint polytope
        h_obs: m_obs x 1 np.ndarray[float]
            The constraint vector of the state constraint polytope

        """
        pass

    @unavailable(not _has_pygame, "pygame")
    def render(self):
        """ Render the environment.

        Any environment should implement _render_env() to perform the actual rendering.
        """
        if '_render_env' not in self.__class__.__dict__:
            print("No rendering implemented")
            return

        self._init_render()
        self._render_env(self.screen, self.axis, self.display_width, self.display_height)
        pygame.display.flip()
        pygame.event.get()
        self._delay_render()

    def collect_metrics(self) -> Dict[str, Any]:
        """Returns metrics from the current episode, if the subclass implements it."""
        return {}

    def _init_render(self):
        if self._render_initialized:
            return

        screen_width = 400
        screen_height = 300
        axis = [-3.0, 3.0, -2.0, 2.0]

        self.screen = pygame.display.set_mode((screen_width, screen_height))
        self.axis = axis
        self.display_width = screen_width
        self.display_height = screen_height

        self.clock = pygame.time.Clock()

        self._render_initialized = True

    def _delay_render(self):
        self.clock.tick(self.delay)

    def _render_env(self, screen, axis: [float], display_width: int, display_height: int):
        """Subclasses should implement this method to render the environment."""
        pass

    def _sample_start_state(self, mean=None, std=None, n_samples=1, normalize=True):
        """ """
        init_std = self.init_std
        if not std is None:
            init_std = std

        init_m = mean
        if init_m is None:
            init_m = self.init_m

        samples = (repmat(init_std, n_samples, 1) * np.random.randn(n_samples, self.n_s) + repmat(init_m, n_samples, 1))

        if normalize:
            samples = samples * self.inv_norm[0]

        return samples.T.squeeze()

    def normalize(self, state=None, action=None):
        """ Normalize the inputs"""
        if not state is None:
            state = self.inv_norm[0] * state

        if not action is None:
            action = self.inv_norm[1] * action

        return state, action

    def unnormalize(self, state=None, action=None):
        """ Unnormalize the inputs"""
        if not state is None:
            state = self.norm[0] * state

        if not action is None:
            action = self.norm[1] * action

        return state, action

    def simulate_onestep(self, state, action):
        """ """

        one_step_dyn = lambda s, t, a: self._dynamics(t, s, a).squeeze()

        # unnormalize state and action
        state = state * self.norm[0]
        action = action.reshape(-1) * self.norm[1]

        sol = odeint(one_step_dyn, state, np.array([0.0, self.dt]), args=(action,))
        next_state = sol[1, :]

        return self.state_to_obs(next_state), self.state_to_obs(next_state, True)

    def linearize_discretize(self, x_center=None, u_center=None, normalize=True):
        """ Discretize and linearize the system around an equilibrium point

        Parameters
        ----------
        x_center: 2x0 array[float], optional
            The linearization center of the state.
            Default: the origin
        u_center: 1x0 array[float], optional
            The linearization center of the action
            Default: zero
        """
        if x_center is None:
            x_center = self.p_origin
        else:
            raise NotImplementedError("For now we only allow linearization at the origin!")

        if u_center is None:
            u_center = np.zeros((self.n_s,))
        else:
            raise NotImplementedError("For now we only allow linearization at the origin!")

        jac_ct = self._jac_dynamics()

        A_ct = jac_ct[:, :self.n_s]
        B_ct = jac_ct[:, self.n_s:]

        if normalize:
            m_x = np.diag(self.norm[0])
            m_u = np.diag(self.norm[1])
            m_x_inv = np.diag(self.inv_norm[0])
            m_u_inv = np.diag(self.inv_norm[1])
            A_ct = np.linalg.multi_dot((m_x_inv, A_ct, m_x))
            B_ct = np.linalg.multi_dot((m_x_inv, B_ct, m_u))

        ct_input = (A_ct, B_ct, np.eye(self.n_s), np.zeros((self.n_s, self.n_u)))
        A, B, _, _, _ = cont2discrete(ct_input, self.dt)

        return A, B

    def step(self, action):
        """ Apply action to system and output current state and other information.

        Parameters
        ----------
        action: n_u x 0 1darray[float]
            The normalized(!) action
        """

        action = self.norm[1] * action  # unnormalize
        action_clipped = np.clip(np.nan_to_num(action), self.u_min, self.u_max)  # clip to normalized max action

        self.odesolver.set_f_params(action)
        old_state = np.copy(self.current_state)
        self.current_state = self.odesolver.integrate(self.odesolver.t + self.dt)
        self.current_episode_trajectory.append(self.current_state)

        self.iteration += 1
        done, result_code = self._check_current_state()

        new_state_noise_obs = self.state_to_obs(np.copy(self.current_state), add_noise=True)
        new_state_obs = self.state_to_obs(np.copy(self.current_state))

        if self.odesolver.successful():

            if self.verbosity > 0:
                print("\n===Old state unnormalized:")
                print(old_state)
                print("===Action unnormalized:")
                print(action)
                print("===Next state unnormalized:")
                print((self.current_state))

            action_clipped_normalized = self.inv_norm[1] * action_clipped
            return action_clipped_normalized, new_state_obs, new_state_noise_obs, done, result_code
        raise ValueError("Odesolver failed!")

    def get_target(self):
        """ Return the target state

        Returns
        -------
        target: n_sx0 1darray[float]
            The target state in observation space
        """
        return self.state_to_obs(self.target)


class InvertedPendulum(Environment):
    """ The Inverted Pendulum environment

    The simple two-dimensional Inverted Pendulum environment.
    The system consists of two states and one action:
    States:
        0. d_theta
        1. theta

    TODO: Need to define a safety/fail criterion
    """

    def __init__(self, name="InvertedPendulum", l=.5, m=.15, g=9.82, b=0., dt=.05, init_m=0., init_std=.01,
                 plant_noise=np.array([0.01, 0.01]) ** 2, u_min=np.array([-1.]), u_max=np.array([1.]),
                 target=np.array([0.0, 0.0]), verbosity=1, norm_x=None, norm_u=None, simple_constraints=True,
                 enable_objectives=False):
        """
        Parameters
        ----------
        name: str, optional
            The name of the system
        l: float, optional
            The length of the pendulum
        m: float, optional
            The mass of the pendulum
        g: float, optional
            The gravitation constant
        b: float, optional
            The friction coefficient of the system
        init_m: 2x0 1darray[float], optional
            The initial state mean
        init_std: float, optional
            The standard deviation of the start state sample distribution.
            Note: This is not(!) the uncertainty of the state but merely allows
            for variation in the initial (deterministic) state.
        u_min: 1x0 1darray[float], optional
            The maximum torque applied to the system
        u_max: 1x0 1darray[float], optional
            The maximum torquie applied to the system
        target: 2x0 1darray[float], optional
            The target state
        """
        super(InvertedPendulum, self).__init__(name, 2, 1, dt, init_m, init_std, plant_noise, u_min, u_max, target,
                                               verbosity)
        self.odesolver = ode(self._dynamics)
        self.l = l
        self.m = m
        self.g = g
        self.b = b
        self.p_origin = np.array([0.0, 0.0])
        self.target = target
        self.target_ilqr = init_m

        self._enable_objectives = enable_objectives
        self._objective_thetas = [-0.1, -0.25, 0.05, -0.2, 0.28, 0.2]
        self._current_objective_index = 0
        self._objective_tolerance = 0.01
        self._current_achieved_objective_states = []
        self._achieved_objectives = []

        warnings.warn("Normalization turned off for now. Need to look into it")
        max_deg = 30
        if norm_x is None:
            norm_x = np.array([1., 1.])  # norm_x = np.array([np.sqrt(g/l), np.deg2rad(max_deg)])

        if norm_u is None:
            norm_u = np.array([1.])  # norm_u = np.array([g*m*l*np.sin(np.deg2rad(max_deg))])

        self.norm = [norm_x, norm_u]
        self.inv_norm = [arr ** -1 for arr in self.norm]

        self._init_safety_constraints(simple_constraints)

    @property
    def l_mu(self) -> ndarray:
        return np.array([0.05, .02])

    @property
    def l_sigm(self) -> ndarray:
        return np.array([0.05, .02])

    def _reset(self):
        self.odesolver.set_initial_value(self.current_state, 0.0)
        self._current_achieved_objective_states = []

    def _check_current_state(self):
        state = self.current_state

        if np.abs(state[1] - self._current_objective) <= self._objective_tolerance:
            self._current_objective_index += 1
            self._achieved_objectives.append(self._current_objective)
            self._current_achieved_objective_states.append(state)
            print(f'Reached objective, advancing to next one '
                  f'({self._current_objective_index}: {self._current_objective})')

        # Check if the state lies inside the safe polytope i.e. A * x <= b.
        res = np.matmul(self.h_mat_safe, state) - self.h_safe.T
        satisfied = not (res > 0).any()
        # We don't use the status code.
        status_code = 0
        return not satisfied, status_code

    def objective_cost_function(self, ps: Tensor) -> Optional[Tensor]:
        if not self._enable_objectives:
            return None

        objective = torch.full_like(ps[:, 1], self._current_objective)
        return torch.abs(objective - ps[:, 1])

    @property
    def _current_objective(self) -> float:
        index = self._current_objective_index % len(self._objective_thetas)
        return self._objective_thetas[index]

    def _dynamics(self, t, state, action):
        """ Evaluate the system dynamics

        Parameters
        ----------
        t: float
            Input Parameter required for the odesolver for time-dependent
            odes. Has no influence in this system.
        state: 2x1 array[float]
            The current state of the system
        action: 1x1 array[float]
            The action to be applied at the current time step

        Returns
        -------
        dz: 2x1 array[float]
            The ode evaluated at the given inputs.
        """

        inertia = self.m * self.l ** 2
        dz = np.zeros((2, 1))
        dz[0] = self.g / self.l * np.sin(state[1]) + action / inertia - self.b / inertia * state[0]
        dz[1] = state[0]

        return dz

    def _jac_dynamics(self):
        """ Evaluate the jacobians of the system dynamics

        Returns
        -------
        jac: 2x3 array[float]
            The jacobian of the dynamics w.r.t. the state and action

        """

        state = np.zeros((self.n_s,))

        inertia = self.m * self.l ** 2
        jac_0 = np.zeros((1, 3))  # jacobian of the first equation (dz[0])
        jac_0[0, 0] = self.b / inertia  # derivative w.r.t. d_theta
        jac_0[0, 1] = self.g / self.l * np.cos(state[1])  # derivative w.r.t. theta
        jac_0[0, 2] = 1 / inertia  # derivative w.r.t. u

        jac_1 = np.eye(1, 3)  # jacobian of the second equation

        return np.vstack((jac_0, jac_1))

    def state_to_obs(self, state=None, add_noise=False):
        """ Transform the dynamics state to the state to be observed

        Parameters
        ----------
        state: 2x0 1darray[float]
            The internal state of the system.
        add_noise: bool, optional
            If this is set to TRUE, a noisy observation is returned

        Returns
        -------
        state: 2x0 1darray[float]
            The state as is observed by the agent.
            In the case of the inverted pendulum, this is the same.

        """
        if state is None:
            state = self.current_state
        noise = 0
        if add_noise:
            noise += np.random.randn(self.n_s) * np.sqrt(self.plant_noise)

        state_noise = state + noise
        state_norm = state_noise * self.inv_norm[0]

        return state_norm

    def plot_current_trajectory(self, axes: Axes) -> bool:
        constraints = Polytope(self.h_mat_safe, self.h_safe)
        constraints.plot(axes, color='lightgrey')

        thetads = [x[0] for x in self.current_episode_trajectory]
        thetas = [x[1] for x in self.current_episode_trajectory]
        axes.plot(thetads, thetas)
        axes.scatter(thetads[0], thetas[0], label='start')

        if self._enable_objectives:
            objective_thetads = [x[0] for x in self._current_achieved_objective_states]
            objective_thetas = [x[1] for x in self._current_achieved_objective_states]
            axes.scatter(objective_thetads, objective_thetas, label='objectives')

        axes.legend()
        axes.set_xlabel('angular velocity (rad/s)')
        axes.set_ylabel('angle to vertical (rad)')

        return True

    def plot_states(self, axes: Axes, states: List[ndarray], includes_initial_samples: bool) -> bool:
        constraints = Polytope(self.h_mat_safe, self.h_safe)
        constraints.plot(axes, color='lightgrey')

        if includes_initial_samples:
            initial_samples = states[0]
            explored_samples = states[1:]
        else:
            initial_samples = None
            explored_samples = states

        if initial_samples is not None:
            initial_thetads = initial_samples[:, 0]
            initial_thetas = initial_samples[:, 1]
            axes.scatter(initial_thetads, initial_thetas, label='initial samples')

        if len(explored_samples) > 0:
            thetads = np.vstack(itertools.chain(explored_samples))[:, 0]
            thetas = np.vstack(itertools.chain(explored_samples))[:, 1]
            axes.scatter(thetads, thetas, label='explored samples')

        axes.legend()
        axes.set_xlabel('angular velocity (rad/s)')
        axes.set_ylabel('angle to vertical (rad)')

        return True

    @unavailable(not _has_matplotlib, "matplotlib")
    def plot_state(self, ax, x=None, color="b", normalize=True):
        """ Plot the current state or a given state vector

        Parameters:
        -----------
        ax: Axes Object
            The axes to plot the state on
        x: 2x0 array_like[float], optional
            A state vector of the dynamics
        Returns
        -------
        ax: Axes Object
            The axes with the state plotted
        """
        if x is None:
            x = self.current_state
            if normalize:
                x, _ = self.normalize(x)
        assert len(x) == self.n_s, "x needs to have the same number of states as the dynamics"
        plt.sca(ax)
        ax.plot(x[0], x[1], color=color, marker="o", mew=1.2)
        return ax

    @unavailable(not _has_matplotlib, "matplotlib")
    def plot_ellipsoid_trajectory(self, p, q, vis_safety_bounds=True, ax=None, color="r"):
        """ Plot the reachability ellipsoids given in observation space

        TODO: Need more principled way to transform ellipsoid to internal states

        Parameters
        ----------
        p: n x n_s array[float]
            The ellipsoid centers of the trajectory
        q: n x n_s x n_s  ndarray[float]
            The shape matrices of the trajectory
        vis_safety_bounds: bool, optional
            Visualize the safety bounds of the system

        """
        new_ax = False

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            new_ax = True

        plt.sca(ax)
        n, n_s = np.shape(p)
        handles = [None] * n
        for i in range(n):
            p_i = cas_reshape(p[i, :], (n_s, 1)) + self.p_origin.reshape((n_s, 1))
            q_i = cas_reshape(q[i, :], (self.n_s, self.n_s))
            ax, handles[i] = plot_ellipsoid_2D(p_i, q_i, ax, color=color)

        if vis_safety_bounds:
            ax = self.plot_safety_bounds(ax)

        if new_ax:
            plt.show()

        return ax, handles

    @unavailable(not _has_matplotlib, "matplotlib")
    def plot_safety_bounds(self, ax=None, plot_safe_bounds=True, plot_obs=False, normalize=True, color=(0., 0., 0.)):
        """ Given a 2D axes object, plot the safety bounds on it

        Parameters
        ----------
        ax: Axes object,
            The input axes object to plot on

        Returns
        -------
        ax: Axes object
            The same Axes object as the input ax but now contains the rectangle
        """

        new_fig = False
        if ax is None:
            new_fig = True
            fig = plt.figure()
            ax = fig.add_subplot(111, aspect='equal')

        if not (plot_safe_bounds or plot_obs):
            warnings.warn("plot_safety_bounds doesn't plot anything")

        x_polygon = self.corners_polygon
        if normalize:
            m_x = np.diag(self.inv_norm[0])
            x_polygon = np.dot(x_polygon, m_x.T)

        if plot_safe_bounds:
            for simplex in self.ch_safety_bounds.simplices:
                ax.plot(x_polygon[simplex, 0], x_polygon[simplex, 1], 'k-')

            # ax.add_patch(mpatch.Polygon(x_polygon,fill = False))
        if new_fig:
            ax.set_xlim(-2., 2.)
            ax.set_ylim(-1., 1.)

            return fig, ax

        return ax

    def get_safe_bounds(self):
        """ Returns the parameters of a rectangle visualizing safety bounds

        Returns
        -------
        p_safe: 2x0 tuple[float]
            The lower left corner of the rectangle representing the safe zone
        width_safe: float
            The width of the safety rectangle
        height_safe: float
            The height of the safety rectangle
        p_safe: 2x0 tuple[float]
            The lower left corner of the rectangle representing the obstacle free zone
        width_safe: float
            The width of the obstacle free rectangle
        height_safe: float
            The height of the obstacle free rectangle
        """
        raise DeprecationWarning("We replace rectangles with polygons. Simpler for drawing")
        dtheta_max_safe = self.h_safe[0]
        dtheta_min_safe = -self.h_safe[1]
        theta_max_safe = self.h_safe[2]
        theta_min_safe = -self.h_safe[3]

        width_safe = dtheta_max_safe - dtheta_min_safe
        height_safe = theta_max_safe - theta_min_safe
        p_safe = (dtheta_min_safe + self.p_origin[0], theta_min_safe + self.p_origin[1])

        return p_safe, width_safe, height_safe

    def random_action(self) -> ndarray:
        c = 0.5
        return c * (np.random.rand(self.n_u) * (self.u_max_norm - self.u_min_norm) + self.u_min_norm)

    def _init_safety_constraints(self, simple_constraints: bool):
        """ Get state and safety constraints

        We define the state constraints as:
            x_0 - 3*x_1 <= 1
            x_0 - 3*x_1 >= -1
            x_1 <= max_rad
            x_1 >= -max_rad
        """

        max_dx = 2.0
        max_deg = 20
        max_dtheta = 1.2
        max_dtheta_theta_0 = 0.8

        max_rad = np.deg2rad(max_deg)

        # -max_dtheta <dtheta <= max_dtheta
        h_0_mat = np.asarray([[1., 0.], [-1., 0.]])
        h_0_vec = np.array([max_dtheta, max_dtheta])[:, None]

        #  (1/.4)*dtheta + (2/.26)*theta <= 1
        # 2*max_dtheta + c*max_rad <= 1
        # => c = (1+2*max_dtheta) / max_rad
        # for max_deg = 30, max_dtheta = 1.5 => c \approx 7.62
        if simple_constraints:
            corners_polygon = np.array([[-max_dtheta_theta_0, max_rad],  #
                                        [max_dtheta_theta_0, max_rad],  #
                                        [max_dtheta_theta_0, -max_rad],  #
                                        [-max_dtheta_theta_0, -max_rad]])
        else:
            corners_polygon = np.array([[-max_dtheta, max_rad],  #
                                        [max_dtheta_theta_0, 0.0],  #
                                        [max_dtheta, -max_rad],  #
                                        [-max_dtheta_theta_0, 0.0]])

        ch = ConvexHull(corners_polygon)

        # returns the equation for the convex hull of the corner points s.t. eq = [H,h]
        # with Hx <= -h
        eq = ch.equations
        h_mat_safe = eq[:, :self.n_s]
        h_safe = -eq[:, self.n_s:]  # We want the form Ax <= b , hence A = H, b = -h

        # normalize safety bounds
        self.h_mat_safe = h_mat_safe
        self.h_safe = h_safe
        self.h_mat_obs = None  # p.asarray([[0.,1.],[0.,-1.]])
        self.h_obs = None  # np.array([.6,.6]).reshape(2,1)

        # arrange the corner points such that it can be ploted via a line plot
        self.corners_polygon = corners_polygon
        self.ch_safety_bounds = ch

    def get_safety_constraints(self, normalize=True):
        """ Return the safe constraints

        Parameters
        ----------
        normalize: boolean, optional
            If TRUE: Returns normalized constraints
        """
        if normalize:
            m_x = np.diag(self.norm[0])
            h_mat_safe = np.dot(self.h_mat_safe, m_x)
        else:
            h_mat_safe = self.h_mat_safe

        return h_mat_safe, self.h_safe, self.h_mat_obs, self.h_obs

    def _render_env(self, screen, axis: [float], display_width: int, display_height: int):
        # Clear screen to black.
        screen.fill((0, 0, 0))

        center_x = display_width / 2
        center_y = display_height / 2

        length = min(display_width, display_height) / 3

        theta = self.current_state[1]
        end_x = center_x - length * math.sin(theta)
        end_y = center_y - length * math.cos(theta)

        pygame.draw.circle(screen, (255, 255, 255), (center_x, center_y), 10)
        pygame.draw.line(screen, (255, 255, 255), (center_x, center_y), (end_x, end_y), width=3)


class CartPole(Environment):
    """ The classic CartPole swing-up Task


    The CartPole swing-up task with the following states:

        ODE:
            0. x-position cart
            1. x-velocity cart
            2. pendulum angle theta
            3. pendulum angle velocity theta

        OBSERVATIONS:
            0. x-position cart
            1. x-velocity cart
            2. pendulum angle theta
            3. pendulum angle velocity theta

    Task: swing up pendulum via the cart in order to reach a upright resting position
    (zero angular velocity)

    """

    def __init__(self, name='CartPole', dt=0.1, l=0.5, m=0.5, M=0.5, b=0.1, g=9.82,
                 init_m=np.array([0.0, 0.0, 0.0, 0.0]), visualize=True, init_std=0.0, u_min=np.array([-4.0]),
                 u_max=np.array([4.0]), norm_x=None, norm_u=None, plant_noise=np.array([0.02, 0.05, 0.02, 0.05]) ** 2,
                 verbosity=1):
        super(CartPole, self).__init__(name, 4, 1, dt, init_m, init_std, plant_noise, u_min, u_max,
                                       np.array([0.0, l, 0.0]), verbosity)

        self.ns_ode = 4

        self.current_state = None

        self.T = 15
        # initialize the physical properties
        self.l = l
        self.m = m
        self.M = M
        self.b = b
        self.g = g
        self.visualize = visualize

        self.idx_angles = np.array([2])
        self.obs_angles_sin = np.array([3])
        self.obs_angles_cos = np.array([4])

        self.target_ilqr = init_m

        self.D_cost = np.array([40, 20, 40])
        self.R_cost = np.array([1.0])

        if norm_x is None:
            x_max = 3.0
            x_dot_max = 5.0
            max_deg = 30
            norm_x = np.array([x_max, x_dot_max, np.deg2rad(max_deg), 2 * np.sqrt(g / l)])

        if norm_u is None:
            norm_u = self.u_max - self.u_min

        self.norm = [norm_x, norm_u]
        self.inv_norm = [arr ** -1 for arr in self.norm]

        self._init_safety_constraints()

        self.odesolver = ode(self._dynamics)
        self.name = name

    @property
    def l_mu(self) -> ndarray:
        return np.array([.05, .05, .05, .05])

    @property
    def l_sigm(self) -> ndarray:
        return np.array([.05, .05, .05, .05])

    def state_to_obs(self, state=None, add_noise=False):
        """ Normalize the state and add observation noise"""

        if state is None:
            state = self.current_state

        obs = np.copy(state)
        noise = 0.

        if add_noise:
            noise += np.random.randn(self.n_s) * np.sqrt(self.plant_noise)

        obs += noise
        obs = obs * self.inv_norm[0]

        return obs

    def _reset(self):
        self.odesolver.set_initial_value(self.current_state, 0.0)

    def _check_current_state(self, state=None) -> Tuple[bool, int]:
        """Checks the constraints.

        For the cartpole environment the error codes are:
            1: obstacle constraint - lim_x violated;
            2: pole fell over - max_rad_theta violated
        """
        status_code = -1

        if state is None:
            state = self.current_state

        sat = True

        if state[0] > self.lim_x[1] or state[0] < self.lim_x[0]:
            sat = False
            status_code = 1

        if -self.max_rad_theta > state[2] or state[2] > self.max_rad_theta:
            sat = False
            status_code = 2

        return not sat, status_code

    def _render_env(self, screen, axis: [float], display_width: int, display_height: int):
        # blacken screen
        screen.fill((0, 0, 0))
        # get the cart box coordinates
        scrwidt = display_width
        scrhght = display_height
        cart_x = self.current_state[0]
        cart_coords = (cart_x - 0.2, cart_x + 0.2, -0.1, 0.1)

        cart_height = float(((cart_coords[3] - cart_coords[2]) / float(axis[3] - axis[2])) * float(scrhght))
        # cart rectangle image coords
        img_coords_cart = [0, 0, 0, 0]
        img_coords_cart[0] = float(
            ((cart_coords[0] - axis[0]) / float(axis[1] - axis[0])) * float(scrwidt))  # left side x coordinate
        img_coords_cart[1] = scrhght - cart_height - float(
            ((cart_coords[3] - axis[2]) / float(axis[3] - axis[2])) * float(scrhght))  # top y coord
        img_coords_cart[2] = float(
            ((cart_coords[1] - cart_coords[0]) / float(axis[1] - axis[0])) * float(scrwidt))  # width
        img_coords_cart[3] = cart_height  # height

        cart_color = (255, 255, 0)

        screen.fill(cart_color, pygame.Rect(tuple(img_coords_cart)))

        cart_coords = (cart_x, 0.0)
        img_coords_pole_0 = self.convert_coords(cart_coords)
        img_coords_pole_1 = self.convert_coords(self._single_pend_top_pos(self.current_state))

        pole_color = (0, 255, 255)
        pygame.draw.line(screen, pole_color, img_coords_pole_0, img_coords_pole_1, 4)

    def plot_ellipsoid_trajectory(self, p, q, vis_safety_bounds=True):
        """Visualize the reachability ellipsoid"""
        raise NotImplementedError()

    def _dynamics(self, t, state, action):
        m = self.m
        M = self.M
        l = self.l
        b = self.b
        g = self.g

        x, v, theta, omega = tuple(np.split(state, [1, 2, 3]))

        det = l * (M + m * np.square(np.sin(theta)))

        dz = np.zeros((4, 1))

        dz[0] = v  # the cart pos
        dz[1] = (action - m * l * np.square(omega) * np.sin(theta) - b * omega * np.cos(
            theta) + 0.5 * m * g * l * np.sin(2 * theta)) * l / det
        dz[2] = omega  # the angle
        dz[3] = (action * np.cos(theta) - 0.5 * m * l * np.square(omega) * np.sin(2 * theta) - b * (m + M) * omega / (
                m * l) + (m + M) * g * np.sin(theta)) / det

        return dz

    def _jac_dynamics(self):
        """ The jacobian of the dynamics at the origin."""

        m = self.m
        M = self.M
        l = self.l
        b = self.b
        g = self.g

        A = np.array([[0, 1, 0, 0],  #
                      [0, 0, .5 * g * m / M, -b * .5 / (M * l)],  #
                      [0, 0, 0, 1],  #
                      [0, 0, g * (m + M) / (l * M), -b * (m + M) / (m * M * l ** 2)]])

        B = np.array([0, 1. / M, 0, 1 / (M * l)]).reshape((-1, self.n_u))

        return np.hstack((A, B))

    def _init_safety_constraints(self):
        """ Get state and safety constraints

        We define the state constraints as:

            x_2 - 3*x_3 <= 1
            x_2 - 3*x_3 >= -1
            x_3 <= max_rad
            x_3 >= -max_rad

        """

        max_deg = 25
        max_dtheta = 1.2
        max_dx = 1.66
        lim_x_safe = [-4, 2.6]

        max_rad = np.deg2rad(max_deg)

        # Safety constraints
        # -max_dtheta <dtheta <= max_dtheta
        h_0_mat = np.asarray([0., 0., 7.25, 1.])[None, :]
        h_0_vec = np.array([1.])[:, None]

        h_1_mat = -h_0_mat
        h_1_vec = h_0_vec

        #  (1/.4)*dtheta + (2/.26)*theta <= 1
        h_2_mat = np.asarray([0., 0., -1.25, -1.])[None, :]
        # h_2_mat = np.asarray([0.,0.,-0.5,-1.])[None,:]
        h_2_vec = np.asarray([1.])[:, None]

        #  (1/.4)*dtheta + (2/.26)*theta  >= -1
        h_3_mat = -h_2_mat
        h_3_vec = h_2_vec

        # d_x <= max_dx
        h_4_mat = np.array([0., 1., 0., 0.])[None, :]
        h_4_vec = np.array([max_dx])[:, None]

        # d_x >= -max_dx
        h_5_mat = -h_4_mat
        h_5_vec = h_4_vec

        # x <= max_x_safe
        h_6_mat = np.array([[1., 0., 0., 0.], [-1., 0., 0., 0.]])
        h_6_vec = np.array([lim_x_safe[1], -lim_x_safe[0]])[:, None]

        h_7_mat = np.array([[1., 2.0, 0., 0.]])
        h_7_vec = np.array([3.0])[:, None]

        # normalize safety bounds
        self.h_mat_safe = np.vstack((h_0_mat, h_1_mat, h_2_mat, h_3_mat, h_4_mat, h_5_mat, h_6_mat, h_7_mat))

        self.h_safe = np.vstack((h_0_vec, h_1_vec, h_2_vec, h_3_vec, h_4_vec, h_5_vec, h_6_vec, h_7_vec))

        # Obstacle

        lim_x_obs = [-10.0, 3.0]
        max_theta_obs = 90
        max_theta_obs = np.deg2rad(max_theta_obs)

        self.lim_x = [-10., 3.0]

        self.max_rad_theta = max_theta_obs

        h_0_mat = np.array([[1., 0., 0., 0.], [-1., 0., 0., 0.]])
        h_0_vec = np.array([lim_x_obs[1], -lim_x_obs[0]])[:, None]

        # x >= -max_x_safe
        h_2_mat = np.array([0., 0., 1., 0.])[None, :]
        h_2_vec = np.array([max_theta_obs])[None, :]

        h_3_mat = -h_2_mat
        h_3_vec = h_2_vec

        self.h_mat_obs = np.vstack((h_0_mat, h_1_mat, h_2_mat, h_3_mat))  # p.asarray([[0.,1.],[0.,-1.]])
        self.h_obs = np.vstack((h_0_vec, h_1_vec, h_2_vec, h_3_vec))  # np.array([.6,.6]).reshape(2,1)

        # arrange the corner points such that it can be ploted via a line plot
        self.corners_polygon = np.array([[-max_dtheta, max_rad],  #
                                         [max_dtheta, 0.0],  #
                                         [max_dtheta, -max_rad],  #
                                         [-max_dtheta, 0.0],  #
                                         [-max_dtheta, max_rad]])

    def get_safety_constraints(self, normalize=True):
        """ Return the safety constraints

        Parameters
        ----------
        normalize: boolean, optional
            If TRUE: Returns normalized constraints

        Returns
        -------
        h_mat_safe: m_safe x n_x np.ndarray[float]
            The constraint matrix of the safety polytope
        h_safe: m_safe x 1 np.ndarray[float]
            The constraint vector of the safety polytope
        h_mat_obs: m_obs x n_x np.ndarray[float]
            The constraint matrix of the state constraint polytope
        h_obs: m_obs x 1 np.ndarray[float]
            The constraint vector of the state constraint polytope

        """
        h_mat_safe = self.h_mat_safe
        h_mat_obs = self.h_mat_obs
        if normalize:
            m_x = np.diag(self.norm[0])
            h_mat_safe = np.dot(h_mat_safe, m_x)
            if not self.h_mat_obs is None:
                h_mat_obs = np.dot(h_mat_obs, m_x)

        return h_mat_safe, self.h_safe, h_mat_obs, self.h_obs

    def random_action(self) -> ndarray:
        return np.random.rand(self.n_u) * (self.u_max_norm - self.u_min_norm) + self.u_min_norm

    def _single_pend_top_pos(self, state):
        """

        """
        # pos_x_y = np.zeros((2,1))
        idx_cartpos = 0

        cart_pos = [state[idx_cartpos], 0.0]
        # cart_pos = vertcat(state[idx_cartpos],0.0)

        sin_ang = np.sin(state[self.idx_angles[0]])
        cos_ang = np.cos(state[self.idx_angles[0]])

        rel_pos_pole1 = [self.l * sin_ang, self.l * cos_ang]

        return np.add(cart_pos, rel_pos_pole1)

    def get_target(self):
        """

        """

        return self.target

    def state_trafo(self, state_obs, variance=None):
        """ State transformation mapping from state space to
            operational (target-) space

        """

        velocity_pend = state_obs[2]
        state_trafo = vertcat(self._single_pend_top_pos(state_obs).reshape((2, 1)), velocity_pend.reshape((1, 1)))
        if variance is None:
            return state_trafo
        else:
            return state_trafo, variance

    def convert_coords(self, coords):
        """
        """
        img_coords = [0, 0]
        img_width = self.display_width
        img_height = self.display_height
        img_coords[0] = float(((coords[0] - self.axis[0]) / float(self.axis[1] - self.axis[0])) * float(img_width))
        img_coords[1] = img_height - float(
            ((coords[1] - self.axis[2]) / float(self.axis[3] - self.axis[2])) * float(img_height))
        return tuple(img_coords)


if __name__ == "__main__":
    pend = CartPole()
    s = pend.reset()
    print(s)

    for i in range(200):
        a = pend.random_action()
        print(a)
        _, s_new, _, _, _ = pend.step(a)
        pend.render()
        print(s_new)

    p = np.vstack((s.reshape((1, -1)), s_new.reshape((1, -1))))
    q = .1 * np.eye(2).reshape((1, -1))
    q = np.stack((q, q))  # pend.plot_ellipsoid_trajectory(p,q,True)
