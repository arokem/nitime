"""

A collection of representations of the hemodynamic response function.

All these functions should have 'duration' as their first input (as a
positional argument. The other arguments are all key-word arguments, standing
for fit parameters of the HRF, with 'Fs' a key-word argumanet determining the
sampling rate. This common interface should help generate flexible fitting
algorithms which would accept any of these as input.

"""

import numpy as np
from scipy.misc import factorial


def gamma(duration, A=1., tau=1.08, n=3, delta=2.05, Fs=1.0):
    r"""A gamma function hrf model, with two parameters, based on
    [Boynton1996]_


    Parameters
    ----------

    duration: float
        the length of the HRF (in the inverse units of the sampling rate)

    A: float
        a scaling factor, sets the max of the function, defaults to 1

    tau: float
        The time constant of the gamma function, defaults to 1.08

    n: int
        The phase delay of the gamma function, defaults to 3

    delta: float
        A pure delay, allowing for an additional delay from the onset of the
        time-series to the beginning of the gamma hrf, defaults to 2.05

    Fs: float
        The sampling rate, defaults to 1.0


    Returns
    -------

    h: the gamma function hrf, as a function of time

    Notes
    -----
    This is based on equation 3 in Boynton (1996):

    .. math::

        h(t) =
        \frac{(\frac{t-\delta}{\tau})^{(n-1)}
        e^{-(\frac{t-\delta}{\tau})}}{\tau(n-1)!}


    Geoffrey M. Boynton, Stephen A. Engel, Gary H. Glover and David J. Heeger
    (1996). Linear Systems Analysis of Functional Magnetic Resonance Imaging in
    Human V1. J Neurosci 16: 4207-4221

    """
    # XXX Maybe change to take out the time (Fs, duration, etc) from this and
    # instead implement this in units of sampling interval (pushing the time
    # aspect to the higher level)?
    if type(n) is not int:
        print ('fmri.hrf.gamma received unusual input, converting n from %s to %i'
               % (str(n), int(n)))

        n = int(n)

    #Prevent negative delta values:
    if delta < 0:
        raise ValueError('in fmri.hrf.gamma, delta cannot be smaller than 0')

    #Prevent cases in which the delta is larger than the entire hrf:
    if delta > duration:
        e_s = 'in fmri.hrf.gamma, delta cannot be larger than the duration'
        raise ValueError(e_s)

    t_max = duration - delta

    t = np.hstack([np.zeros((delta * Fs)), np.linspace(0, t_max, t_max * Fs)])

    t_tau = t / tau

    h = (t_tau ** (n - 1) * np.exp(-1 * (t_tau)) /
         (tau * factorial(n - 1)))

    return A * h / max(h)

def diff_of_gammas(duration, delta=2.0, A1=1.0, tau1=1.1, n1=5,
                   A2=0.4, tau2=0.9, n2=12, Fs=1.0):
    """

    HRF function based on the difference of two gammas, one for the upshoot and
    one for the subsequent undershoot. Based loosely on Glover (1999). Note
    that this implementation is literally a difference of two gamma functions,
    with one common delay.

    Parameters
    ----------

    duration: float
        The length of the HRF produced, in units of 1/Fs

    delta: float
        The hemodynamic delay, in units of 1/Fs

    A1, A2: float, optional
       These parameterize the amplitude of the first and second gamma,
       respectively.

    tau1, tau2: float, optional
       These parameterize the time-course of the first and second gamma,
       respectievely.

    n1, n2: int, optional
       Parameterize the phase delay of the two gammas

    Fs: float
       The sampling rate (presumably in Hz).

    Glover GH (1999) Deconvolution of Impulse Response in Event-Related BOLD
    fMRI. NeuroImage 9: 416 - 429.

    """

    return (gamma(duration, A1, tau1, n1, delta, Fs) -
            gamma(duration, A2, tau2, n2, delta, Fs))


def two_gamma(duration, a1=6.0, b1=0.9, a2=12.0, b2=0.9, c=0.35, Fs=1.0):
    """

    The canonical two-gamma HRF, based on Glover (1999), but taken directly
    from Harvey and Dumoulin (2011) and the defaults are taken from the methods
    section of that paper.

    Parameters
    ----------

    duration: float
        In units of 1/Fs

    a1, a2: float
        Time-scale parameters for the upswing and post-undershoot respectively

    b1, b2:


    c:

    Returns
    -------

    Note
    ----

    This implements:
    ..math::

        h(t) = (t/d_1)^{a_1} e^{(-t-d_1)/b_1} - c(t/d_2)^{a_2} e (-(t-d_2)/b2)

    Note that, to constrain the degrees of freedom in fitting this function, we
    raise and error whenever d1 > d2 and whenever c < 0.

    Glover GH (1999) Deconvolution of Impulse Response in Event-Related BOLD
    fMRI. NeuroImage 9: 416 - 429.

    Harvey BM and Dumoulin SO (2011). The relationship between cortical
    magnification factor and population receptive field size in human visual
    cortex: constancies in cortical architecture. J Neurosci: 31: 13604-13612

    """

    d1 = a1 * b1
    d2 = a2 * b2
    if d1 >= d2:
        raise ValueError("d1 = %s, d2 = %s, but must have d1 < d2"%(d1,d2))
    if c < 0:
        raise ValueError("c = %s, but must be smaller than 0"%c)

    t = np.linspace(0, duration, duration * Fs)

    return ((t/d1)**a1 * np.exp(-(t-d1)/b1) -
            c * (t/d2)**a2 * np.exp(-(t-d2)/b2))



def two_sin(duration, A, B, tau1, f1, tau2, f2, Fs=1.0):
    r"""

    HRF based on Polonsky (2000):

    .. math::

       H(t) = exp(\frac{-t}{\tau_1}) sin(2\cdot\pi f_1 \cdot t) -a\cdot
       exp(-\frac{t}{\tau_2})*sin(2\pi f_2 t)

    Alex Polonsky, Randolph Blake, Jochen Braun and David J. Heeger
    (2000). Neuronal activity in human primary visual cortex correlates with
    perception during binocular rivalry. Nature Neuroscience 3: 1153-1159

    """
    sampling_interval = 1 / float(Fs)

    t = np.arange(0, t_max, sampling_interval)

    h = (np.exp(-t / tau1) * np.sin(2 * np.pi * f1 * t) -
            (B * np.exp(-t / tau2) * np.sin(2 * np.pi * f2 * t)))

    return A * h / max(h)
