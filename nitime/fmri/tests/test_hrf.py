import numpy as np
import numpy.testing as npt

import nitime.fmri.hrf as hrf

def test_hrf():
    duration = 33
    h = []
    for this_hrf in [hrf.gamma, hrf.diff_of_gammas, hrf.two_gamma, hrf.two_sin]:
        # This makes sure that the common interface is kept (everything's a
        # kwarg except for the duration):
        h.append(this_hrf(duration))
