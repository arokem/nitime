"""

Generate a variety of PRF (or population receptive field) models, of the kind described in
Dumoulin and Wandell (2008).

Dumoulin and Wandell (2008). Population receptive field estimates in human
visual cortex. Neuroimage 39: 647-660.

"""

import numpy as np

# Set some globals:
default_nx = 256
default_ny = 256
default_sd = 20

def _mk_meshgrid(n_x=default_nx, n_y=default_nx, x0=default_nx/2, y0=default_ny/2):
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

def gaussian(n_x=default_nx, n_y=default_ny, x0=default_nx/2, y0=default_ny/2,
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
         x and y dimentsions

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


def sin_grating(n_x=default_nx, n_y=default_ny, x0=default_nx/2,
                y0=default_ny/2, sf=0.1, ori=0, phase=0):
    """

    Create an n_x by n_y image with a sin grating with the phase relative to
    x0, y0, spatial frequency sf and orientation ori (set in degrees)

    """
    rad = np.deg2rad(ori)

    x, y = _mk_meshgrid(n_x, n_y, x0, y0)
    return np.sin(sf * (x * np.cos(rad) - y * np.sin(rad)) + phase )


def cos_grating(n_x=default_nx, n_y=default_ny, x0=default_nx/2,
                y0=default_ny/2, sf=0.1, ori=0, phase=0):
    """
    Create an n_x by n_y image with a sin grating with the phase relative to
    x0, y0, spatial frequency sf and orientation ori (set in degrees)

    """

    rad = np.deg2rad(ori)

    x, y = _mk_meshgrid(n_x, n_y, x0, y0)
    return np.cos(sf * (x * np.cos(rad) - y * np.sin(rad)) + phase )

def gabor(n_x=default_nx, n_y=default_ny, x0=default_nx/2, y0=default_ny/2,
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

def response(stim,p,h):
    """

    estimate a response given a stimulus, prf, hrf combination. This will be
    used for the fitting procedure

    """
    neural_response = np.array(np.matrix(stim) * np.matrix(p).T).squeeze()
    bold_response = np.convolve(neural_response, h)
