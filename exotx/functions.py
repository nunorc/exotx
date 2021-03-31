
""" Functions for performing several tasks. """

from typing import List
import logging

import lightkurve as lk
import numpy as np
from scipy import signal
from astropy.time import Time

from .classes import Args, LightCurve, Params, Fit, Transit, Periodogram

logger = logging.getLogger(__name__)

def retrieve(args: Args) -> List[LightCurve]:
    """ Retrieve a collection of lightcurves.

    Retrieve a collection of light curves.
    This is a wrapper for [lightkurve.search_lightcurve](https://docs.lightkurve.org/reference/api/lightkurve.search_lightcurve.html?highlight=search_lightcurve).

    Args:
        args (Args): a instance of the Args class

    Returns:
        lcc: a list of light curves
    """
    quarters = list(args.quarters)

    lcc = lk.search_lightcurve(args.target,
                               author='Kepler',  # TODO: get from Args
                               quarter=quarters,
                               cadence=args.cadence).download_all()

    return list(lcc)

def detrend(lc: LightCurve) -> LightCurve:
    """ Detrend a lightcurve.

    Removes the low frequency trend using scipy’s Savitzky-Golay filter.
    This is a wrapper for [lightkurve.LightCurve.flatten](https://docs.lightkurve.org/reference/api/lightkurve.LightCurve.flatten.html#lightkurve.LightCurve.flatten).

    Args:
        lc (LightCurve): a lightcurve

    Returns:
        lc: a detrended lightcurve
    """
    result = None

    if lc:
        result = lc.flatten()

    return result

def normalize(lc: LightCurve) -> LightCurve:
    """ Normalize a lightcurve.

    Divides the flux and the flux error by the mean value.
    This is a wrapper for [lightkurve.LightCurve.normalize](https://docs.lightkurve.org/reference/api/lightkurve.LightCurve.normalize.html#lightkurve.LightCurve.normalize).

    Args:
        lc (LightCurve): a lightcurve

    Returns:
        lc: a normalized lightcurve
    """
    result = None

    if lc:
        result = lc.normalize()

    return result

def combine(lcs: List[LightCurve]) -> LightCurve:
    """ Combine a list of lightcurves.

    Append together a list of light curves.

    Args:
        lcs: a list of LightCurve objects

    Returns:
        lc: a combined LightCurve object
    """
    result = None

    if len(lcs) > 1:
        curr = lcs[0]
        for i in lcs[1:]:
            curr = curr.append(i)
        result = curr
    else:
        result = lcs[0]

    return result

def periodogram(lc: LightCurve, **kwargs) -> Periodogram:
    """ Build a periodogram for a light curve.

    TODO: details on building periodrograms from the lightkurve package

    Args:
        lc: a LightCurve object

    Returns:
        pg: a Periodogram object
    """
    pg = None

    if lc:
        pg = lc.to_periodogram(method='bls', **kwargs)

    return pg

def find_pg_params(pg: Periodogram, n: int = 10) -> List[Params]:
    """ Find peaks of interest in a periodogram object.

    TODO: more details

    Args:
        pg: a Periodogram object

    Returns:
        params: a list of Params objects
    """
    result = []

    peaks, _ = signal.find_peaks(pg.power, prominence=int(pg.power.max()/10))

    params = []
    for idx in peaks:
        params.append(Params({ 'power': pg.power[idx].value,
                               'period': pg.period[idx].value,
                               't0': pg.transit_time[idx].value }))
    params = sorted(params, key=lambda i: i.power, reverse=True)

    if len(params) > 0:
        result.append(params[0])
        i = 1
        while (i < len(params)) and (len(result) < n):
            p1, p2 = int(result[-1].period), int(params[i].period)
            i += 1
            if (p1 < 1.0) or (p2 < 1.0) or (p2 % p1 == 0) or (p1 % p2 == 0):
                continue
            result.append(params[i-1])

    return result

def fold(lc: LightCurve, params: Params,
         phase: float = 0.5, keep_time: bool = False) -> List[LightCurve]:
    """ Fold a light curve.

    TODO: more details

    Args:
        lc: a Periodogram object
        params: a Params object
        phase: the phase
        keep_time: keep original time or set to phase

    Returns:
        lcs: a list LightCurve objects (the folds)
    """
    folds = []

    tmid = params.t0
    while tmid < lc.time[-1].value:
        tstart = Time(tmid-phase, format=lc.time.format)
        tend = Time(tmid+phase, format=lc.time.format)

        try:
            curr = lc.loc[tstart:tend]

            if len(curr) > 0:
                if not keep_time:
                    curr.time = curr.time - tmid
                folds.append(curr)
        except:
            logger.warning('Missed light curve in fold.')

        tmid += params.period

    return folds

def fit_continuum(lc: LightCurve, phase: float = 0.2) -> Fit:
    """ Fit the continuum for a light curve.

    TODO: more details

    Args:
        lc: a LightCurve object
        phase: the phase of the transit datapoints
        keep_time: keep original time or set to phase

    Returns:
        lcs: a list LightCurve objects (the folds)
    """
    tstart = -phase
    tend = phase

    if not isinstance(lc, LightCurve):
        return None

    mask = np.logical_or(lc.time.value < tstart, lc.time.value > tend)
    if np.any(mask) > 0:
        ts = lc.time.value[mask]
        ys = lc.flux.value[mask]

    coefs = np.polyfit(ts, ys, 1)

    return Fit({ 'method': 'polynomial', 'degree': 1, 'coefs': coefs })

def normalize_t(lc: LightCurve, fit: Fit) -> Transit:
    """ Normalizer a light curve given a continuum fit.

    TODO: more details

    Args:
        lc: a LightCurve object
        fit: a Fit object

    Returns:
        transit: a LightCurve object
    """
    lc_normalized = None

    if lc and isinstance(lc, LightCurve):
        lc_normalized = lc.copy()
        poly = np.polyval(fit.coefs, lc.time.value)
        lc_normalized.flux = lc.flux.value / poly

    return lc_normalized
