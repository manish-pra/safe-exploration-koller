# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 16:12:30 2017

@author: tkoller
"""
import os.path

import numpy as np
import pytest
from casadi import MX, Function
from casadi import reshape as cas_reshape

try:
    from safe_exploration.ssm_gpy.gaussian_process import SimpleGPModel
    from GPy.kern import RBF
    _has_ssm_gpy = True
except:
    _has_ssm_gpy = False

from .. import gp_reachability as reach_num
from .. import gp_reachability_casadi as reach_cas
from ..utils import array_of_vec_to_array_of_mat

a_tol = 1e-5
r_tol = 1e-4


# @pytest.fixture(params = [("InvPend",True,True)])
@pytest.fixture(params=[("InvPend", True, True), ("InvPend", False, True),
                        ("InvPend", True, True), ("InvPend", False, True)])
def before_test_onestep_reachability(request):
    if not _has_ssm_gpy:
        pytest.skip("Test requires optional dependencies 'ssm_gpy'")

    np.random.seed(125)

    env, init_uncertainty, lin_model = request.param
    n_s = 2
    n_u = 1
    c_safety = 2
    a = None
    b = None
    if lin_model:
        a = np.random.rand(n_s, n_s)
        b = np.random.rand(n_s, n_u)

    train_data = np.load(os.path.join(os.path.dirname(__file__), 'invpend_data.npz'))
    X = train_data["X"]
    y = train_data["y"]
    m = 50
    #kerns = [RBF(n_s+n_u)]*n_s
    gp = SimpleGPModel(n_s, n_s, n_u, X, y, m, train=False)
    gp.train(X, y, m, opt_hyp=True, choose_data=False)
    L_mu = np.array([0.001] * n_s)
    L_sigm = np.array([0.001] * n_s)
    k_fb = np.random.rand(n_u, n_s)  # need to choose this appropriately later
    k_ff = np.random.rand(n_u, 1)

    p = .1 * np.random.randn(n_s, 1)
    if init_uncertainty:
        q = .2 * np.array(
            [[.5, .2], [.2, .65]])  # reachability based on previous uncertainty
    else:
        q = None  # no initial uncertainty

    return p, q, gp, k_fb, k_ff, L_mu, L_sigm, c_safety, a, b


def test_onestep_reachability(before_test_onestep_reachability):
    """ do we get the same results as the numpy equivalent?"""

    p, q, gp, k_fb, k_ff, L_mu, L_sigm, c_safety, a, b = before_test_onestep_reachability

    n_u, n_s = np.shape(k_fb)

    k_fb_cas = MX.sym("k_fb", (n_u, n_s))
    k_ff_cas = MX.sym("k_ff", (n_u, 1))

    p_new_cas, q_new_cas, _ = reach_cas.onestep_reachability(p, gp, k_ff_cas, L_mu,
                                                             L_sigm, q, k_fb_cas,
                                                             c_safety, a=a, b=b)
    f = Function("f", [k_fb_cas, k_ff_cas], [p_new_cas, q_new_cas])

    f_out_cas = f(k_fb, k_ff)

    f_out_num = reach_num.onestep_reachability(p, gp, k_ff, L_mu, L_sigm, q, k_fb,
                                               c_safety, 0, a=a, b=b)

    assert np.allclose(f_out_cas[0],
                       f_out_num[0]), "Are the centers of the next state the same?"
    assert np.allclose(f_out_cas[1], f_out_num[
        1]), "Are the shape matrices of the next state the same?"


# @pytest.mark.xfail
def test_multistep_reachability(before_test_onestep_reachability):
    """ """
    p, _, gp, k_fb, _, L_mu, L_sigm, c_safety, a, b = before_test_onestep_reachability
    T = 3

    n_u, n_s = np.shape(k_fb)

    u_0 = .2 * np.random.randn(n_u, 1)
    k_fb_0 = np.random.randn(T - 1,
                             n_s * n_u)  # np.zeros((T-1,n_s*n_u))# np.random.randn(T-1,n_s*n_u)
    k_ff = np.random.randn(T - 1, n_u)
    # k_fb_ctrl = np.zeros((n_u,n_s))#np.random.randn(n_u,n_s)

    u_0_cas = MX.sym("u_0", (n_u, 1))
    k_fb_cas_0 = MX.sym("k_fb", (T - 1, n_u * n_s))
    k_ff_cas = MX.sym("k_ff", (T - 1, n_u))

    p_new_cas, q_new_cas, _ = reach_cas.multi_step_reachability(p, u_0, k_fb_cas_0,
                                                                k_ff_cas, gp, L_mu,
                                                                L_sigm, c_safety, a, b)
    f = Function("f", [u_0_cas, k_fb_cas_0, k_ff_cas], [p_new_cas, q_new_cas])

    k_fb_0_cas = np.copy(k_fb_0)  # np.copy(k_fb_0)

    for i in range(T - 1):
        k_fb_0_cas[i, None, :] = k_fb_0_cas[i, None, :] + cas_reshape(k_fb,
                                                                      (1, n_u * n_s))
    p_all_cas, q_all_cas = f(u_0, k_fb_0_cas, k_ff)

    k_ff_all = np.vstack((u_0.T, k_ff))

    k_fb_apply = array_of_vec_to_array_of_mat(k_fb_0, n_u, n_s)

    for i in range(T - 1):
        k_fb_apply[i, :, :] += k_fb

    _, _, p_all_num, q_all_num = reach_num.multistep_reachability(p, gp, k_fb_apply,
                                                                  k_ff_all, L_mu,
                                                                  L_sigm, None,
                                                                  c_safety, 0, a, b,
                                                                  None)

    assert np.allclose(p_all_cas, p_all_num, r_tol,
                       a_tol), "Are the centers of the ellipsoids same?"

    assert np.allclose(q_all_cas[0, :], q_all_num[0, :, :].reshape((-1, n_s * n_s)),
                       r_tol, a_tol), "Are the first shape matrices the same?"
    assert np.allclose(q_all_cas[1, :], q_all_num[1, :, :].reshape((-1, n_s * n_s)),
                       r_tol, a_tol), "Are the second shape matrices the same?"
    # assert np.allclose(q_all_cas[1,:],q_all_num[1,:].reshape((-1,n_s*n_s))), "Are the second shape matrices the same?"
    assert np.allclose(q_all_cas[-1, :], q_all_num[-1, :, :].reshape((-1, n_s * n_s)),
                       r_tol, a_tol), "Are the last shape matrices the same?"
    assert np.allclose(q_all_cas, q_all_num.reshape((T, n_s * n_s)), r_tol,
                       a_tol), "Are the shape matrices the same?"
