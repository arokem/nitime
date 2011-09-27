import numpy as np
import numpy.testing as npt

import nitime.fmri.prf as prf
import nitime.fmri.hrf as hrf

def test_norms():
    """

    Test norm-generating functions

    Based on the example in http://mathworld.wolfram.com/VectorNorm.html
    """

    v1 = [1,2,3]
    l1_1 = prf.l1_norm(v1)
    l2_1 = prf.l2_norm(v1)

    npt.assert_equal(l1_1,6)
    npt.assert_equal(l2_1,np.sqrt(14))

    v2 = np.array([[1,2,3],
                  [4,5,6]])

    l1_2 = prf.l1_norm(v2)
    l2_2 = prf.l2_norm(v2)
    npt.assert_equal(l1_2, 21)
    npt.assert_equal(l2_2, np.sqrt(91))

def test_unit_length():

    """

    Test unit-length normalization for different vector norms

    """

    for norm in [prf.l1_norm, prf.l2_norm]:
        # For 1-D arrays:
        v1 = np.random.randn(100)
        result = prf.unit_length(v1,norm)
        npt.assert_almost_equal(norm(result), 1.0)

        # For 2-D arrays:
        v2 = np.random.rand(100,100)
        result = prf.unit_length(v2, norm)
        npt.assert_almost_equal(norm(result), 1.0)


def test_matrix_ops():
    """

    Test the generation of polynomial_matrix and operations on this one.

    """

    x = np.sort(np.random.randn(100,1), 0)
    pm = prf.polynomial_matrix(100, range(3))
    m = prf.projection_matrix(pm)
    x2 = m * x


def test_response():
    """

    Response function (to be compared to the actual data)

    """

    # The dimensions of the PRF/stimulus. Let's try something anisotropic:
    n = 100
    m = 110
    t = 180

    stim = np.random.randn(n,m,t)
    p = prf.gabor(n,m)

    # The simplest call, no static non-linearity, no HRF:
    r1 = prf.response(stim,p)

    # Static non-linearity, but no hrf:
    r2 = prf.response(stim,p,stat_non_lin=prf.exponent,non_lin_params=dict(n=0.5))

    # HRF function, but no static non-linearity:
    r3 = prf.response(stim,p,h=hrf.gamma, h_dur=33, h_params=dict(tau=1.5, Fs=0.5))

    # HRF vector and no static non-linearity:
    r4 = prf.response(stim,p,h=hrf.gamma(33,tau=1.5,Fs=0.5))

    # These two last ones should be equal:
    npt.assert_equal(r3,r4)

    # The full monty:
    r5 = prf.response(stim,p,
                      h=hrf.gamma, h_dur=33, h_params=dict(tau=1.5, Fs=0.5),
                      stat_non_lin=prf.exponent, non_lin_params=dict(n=0.5))

def test_errfunc():
    """
    Testing that the error-function does what it's supposed to.

    """
    stim_dur = 180
    stim = np.random.randn(100,100,stim_dur)
    real_p = prf.gaussian(100,100)
    real_h = hrf.gamma(33)
    bold = prf.response(stim,real_p,real_h)

    e = prf.err_func(bold, stim, prf.gaussian, hrf.gamma)

    # In this case, the prf and hrf are right on, so there should be no error:
    npt.assert_equal(np.all(e==0),True)

    # Now move the PRF slightly away:
    real_p = prf.gaussian(100,100,x0=60)
    bold = prf.response(stim,real_p,real_h)

    e = prf.err_func(bold, stim, prf.gaussian, hrf.gamma)
