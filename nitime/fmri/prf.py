"""

Generate a variety of PRF (or population receptive field) models, of the kind described in
Dumoulin and Wandell (2008) and fit them to data.

Dumoulin and Wandell (2008). Population receptive field estimates in human
visual cortex. Neuroimage 39: 647-660.

"""

import numpy as np
import scipy.linalg as la

# Set some globals:
default_nx = 256
default_ny = 256
default_sd = 20

def _mk_meshgrid(n_x=default_nx, n_y=default_nx, x0=None, y0=None):
    """
    Helper function used to generate meshgrids for subsequenct calculation of
    various filters over the entire grid.

    Parameters
    ----------

    n_x, n_y: int, optional
       The size of the grid in each dimension. Defaults to the global default
       of the module

    x0, y0: int, optional.
       The location of the origin in each dimension. Defaults to the center of
       the grid.

    Returns
    -------
    x,y : a tuple of float/int arrays with x and y values of the grids

    """
    if x0 is None:
        x0 = n_x/2
    if y0 is None:
        y0 = n_y/2

    return np.meshgrid(np.arange(-x0,n_x-x0),np.arange(-y0,n_y-y0))

def _dist_params(ori=0, sd_x=default_sd, sd_y=default_sd):
    """
    Helper function to get the params needed for 2d distribution functions
    (such as the gaussian).

    Parameters
    ----------

    ori: The orientation of the distribution, relative to the xy axis.

     sd_x, sd_y: The standard deviation of the distribution on each dimension.

    """
    rad_rot = np.deg2rad(ori)

    a = ((np.cos(rad_rot) ** 2) / (2 * sd_x**2) +
         (np.sin(rad_rot) ** 2) / (2 * sd_y**2))

    b = (-(np.sin(2 * rad_rot) / (4 * sd_x**2)) +
          (np.sin(2 * rad_rot) / (4 * sd_y**2)))

    c = ((np.sin(rad_rot) ** 2) / (2 * sd_x**2) +
         (np.cos(rad_rot) ** 2) / (2 * sd_y**2))

    return a,b,c

def gaussian(n_x=default_nx, n_y=default_ny, x0=None, y0=None,
             sd_x=default_sd, sd_y=default_sd, ori=0, height=1.0):
    """

    Create an n_x by n_y image of a Gaussian distribution
    centered on x0, y0 with the given standard deviations (in each dimension,
    with a rotation applied, if required)
    and height

    Parameters
    ----------
    n_x, n_y: The size of the Gaussian in the x and y dimensions

    x0, y0: The parameters setting the location of the distribution mode in the
         x and y dimentsions. Defaults to the center.

    sd_x, sd_y: The parameters setting the dispersion parameters (standard
         deviation) of the distribution in the x and y dimensions.

    ori: float,
        The orientation of the distribution relative to the xy axes. Default
        to 0

    height: Sets the maximum value of the function

    Returns
    -------
    g: 2d array with the values of the gaussian pdf in each location.

    Notes
    -----
    Based on: http://en.wikipedia.org/wiki/Gaussian_function

    """
    x, y = _mk_meshgrid(n_x, n_y, x0, y0)

    a,b,c = _dist_params(ori, sd_x, sd_y)

    return height * np.exp(-(a * x**2 + 2.0 * b * x * y + c * y**2))


def sin_grating(n_x=default_nx, n_y=default_ny, x0=None,
                y0=None, sf=0.1, ori=0, phase=0):
    """

    Create an n_x by n_y image with a sin grating with the phase relative to
    x0, y0, spatial frequency sf and orientation ori (set in degrees)

    """
    rad = np.deg2rad(ori)

    x, y = _mk_meshgrid(n_x, n_y, x0, y0)
    return np.sin(sf * (x * np.cos(rad) - y * np.sin(rad)) + phase )


def cos_grating(n_x=default_nx, n_y=default_ny, x0=None,
                y0=None, sf=0.1, ori=0, phase=0):
    """
    Create an n_x by n_y image with a sin grating with the phase relative to
    x0, y0, spatial frequency sf and orientation ori (set in degrees)

    """

    rad = np.deg2rad(ori)

    x, y = _mk_meshgrid(n_x, n_y, x0, y0)
    return np.cos(sf * (x * np.cos(rad) - y * np.sin(rad)) + phase )

def gabor(n_x=default_nx, n_y=default_ny, x0=None, y0=None,
          ori=0, sd=default_sd, sf=0.1, phase=0, carrier=sin_grating):
    """

    A Gabor filter (A multiple of a sin/cos with a Gaussian).

    Parameters
    ----------
    n_x, n_y: int
        The size of the resulting grid.

    x0, y0: int
        The location of the center of the Gabor, relative to the grid

    ori: float
        The orientation of the Gabor (in degrees).

    sd: float
        The standard deviation of the Gaussian envelope.

    sf: float
        The spatial frequency of the carrier

    phase: float
        The phase of the carrier.

    carrier: callable
        Either sin_grating or cos_grating

    Returns
    -------

    2d array with a Gabor filter

    """
    wave = carrier(n_x, n_y, x0, y0, sf, ori, phase)
    gauss = gaussian(n_x, n_y, x0, y0, sd_x=sd, sd_y=sd)

    return gauss * wave

def response(stim, p, h=None, h_dur=None, h_params=None, stat_non_lin=None,
             non_lin_params=None):
    """

    estimate a response given a stimulus, prf, hrf combination. This will be
    used for the fitting procedure

    Parameters
    ----------

    stim: float array, with shape being either (n,m,t) or (n**2,t)
       The stimulus displayed

    p: float array with shape either (n,m) or n**2
       The population receptive field

    h: array or callable
       A parameterized response function, through which the response is
       convolved. Typically, this is the hemodynamic response function of a
       voxel for which we are fitting.

    h_dur: float, optional
       The duration of h, if this is a callable.

    stat_non_lin: callable, optional
       A static non-linearity to apply to the

    non_lin_params: dict, optional
       Parameters to pass to the static non-linearity as parameters

    kwargs: key-word argument parameters to the HRF function, if this is a callable.

    Fs: The sampling rate

    Notes
    -----

    The operation performed by this function is:

         A * B = stat_non_lin((STIM X PRF)) * HRF

    Where A is an estimated response of a neural element with the receptive
    field PRF to the time-varying stimulus STIM, subsequently convolved by a
    response output function HRF

    """

    if stat_non_lin is None:
        # The function will simply return it's input. Allow for passing of
        # kwargs, so that one call fits all situations, but this is moot:
        def stat_non_lin(x, kwargs): return x

    if non_lin_params is None:
        # Just make up a dummy dict:
        non_lin_params = dict(kwargs=None)

    # Reshape these, before performing the cross-multiplication:
    # For stimulus, time is the last dimension and remains:
    stim_re = np.matrix(stim.reshape(np.prod(stim.shape[:-1]),stim.shape[-1]))
    # For the PRF, both dims are space, so we ravel:
    p_re = np.matrix(p.ravel())

    # Cross-multiply the matrices and apply the static non-linearity:
    neural_response = stat_non_lin(np.array(stim_re.T * p_re.T).squeeze(),
                                   **non_lin_params)

    # If no h function/vector is provided, the output is just the neural response:
    if h is None:
        bold_response = neural_response
    else:
        # If it is a function, we need to generate the vector of values for the
        # convolution:
        if callable(h):
            h = h(h_dur, **h_params)

        # Now do the convolution with the result of that operation:
        bold_response = np.convolve(neural_response, h)

    return bold_response

def l2_norm(x):
    """
    The L2 norm of the vector x defined as:

    .. math::

        \sqrt{||x||}

    Parameters
    ----------
    x: array

    Returns
    -------
    float: the L2 norm of x

    Notes
    -----

    See: http://mathworld.wolfram.com/VectorNorm.html

    """
    return np.sqrt(np.dot(np.asarray(x).ravel(), np.asarray(x).ravel()))

def l1_norm(x):
    """
    The L1 norm of the vector x defined as:

    .. math::

       \sum_{i=1}^{n}{|x_i|}

    """
    return np.sum(np.abs(np.asarray(x).ravel()))

def unit_length(x, norm=l2_norm):
    """
    Normalize the input x, relative to a vector norm function.

    Parameters
    ----------

    x: array

    norm: callable (optional)
        A norm function over x, default: l2_norm

    Returns
    -------
    array: x/norm(x)

    """

    return x / norm(x)

def polynomial_matrix(n, degrees):
    """
    Construct a

    """
    f = []

    x = np.linspace (-1,1,n)
    for d in degrees:
        f.append(x**d)

    return np.array(f).T

def ols_matrix(A, norm=None):
    """
    Generate the matrix used to solve OLS regression.

    Parameters
    ----------

    A: float array
        The design matrix

    norm: callable, optional
        A normalization function to apply to the matrix, before extracting the
        OLS matrix.

    Notes
    -----

    The matrix needed for OLS regression for the equation:

    ..math ::

        y = Ax

   is given by:

    ..math ::

        OLS = (A x A')^{-1} x A'

    """
    if norm is not None:
        X = norm(np.matrix(A.copy()))
    else:
        X = np.matrix(A.copy())

    return la.inv(X.T * X) * X.T


def projection_matrix(A, norm=None):
    """
    This is the matrix used to 'project out' a set of variables

    Parameters
    ----------
    A : array
        The design matrix of the variables projected out

    Returns
    -------
    array: the matrix used to project out the variables.
         This operation is achieved by subtracting out the OLS matrix of the set of
         variables in the design matrix:

         .. math ::

             P x y = y - A x (A x A')^{-1} x A' x y  = (I - A (A x A')^{-1}) x y

    """
    return np.eye(A.shape[0]) - A * ols_matrix(A, norm)


def exponent(x, n=None):
    """

    A simple static non-linearity for :func:`response`

    """
    if n is None:
        return x
    else:
        return x ** n


def err_func(bold, stim, prf, hrf, prf_params=None, hrf_params=None,
             extra_params=None, non_lin=None, non_lin_params=None):
    """
    This is the error-function to optimize on.

    Given BOLD data from a voxel, prf starting params and hrf starting params,
    find the best fit params for these factors.

    bold, stim are given, but the rest should be fit.

    """

    p = prf(n_x=stim.shape[0], n_y=stim.shape[1])

    if callable(hrf):
        h = hrf(33)

    r = response(stim, p, h=h, h_dur=33,
                 h_params=hrf_params, stat_non_lin=non_lin,
                 non_lin_params=non_lin_params)

    # This is the error:
    return r - bold
