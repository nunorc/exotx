
""" exotx init """

import logging

from .classes import Args, LightCurve, Transit, Periodogram, Params, Fit
from .functions import retrieve, detrend, normalize, combine, \
                       periodogram, find_pg_params, fold, fit_continuum, normalize_t
from .plotting import plot_lc, plot_folds

logging.basicConfig(format='%(asctime)s | %(levelname)s: %(message)s', level=logging.WARNING)
