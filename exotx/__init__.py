
""" exotx init """

import logging

from .classes import Args, LightCurve, Transit, Periodogram, Params, Fit
from .functions import retrieve, detrend, normalize, combine, \
                       periodogram, find_pg_params, fold, fit_continuum, normalize_t
from .plotting import plot_lc, plot_folds
from .models import LightCurveModel

logging.basicConfig(format = '%(asctime)s | %(levelname)s: %(message)s',
                    datefmt = "%Y-%m-%d %H:%M:%S",
                    level = logging.WARNING)
