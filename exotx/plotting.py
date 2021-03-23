
""" Auxiliary functions for quickly plotting. """

import logging

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

def plot_lc(lc, err=True, fit=None, figsize=(12,6), xlabel='Time', ylabel='Flux'):
    """ Plot a lightcurve.

    Args:
        lc: a LightCurve object
    """
    _, ax = plt.subplots(figsize=figsize)

    if err:
        ax.errorbar(lc.time.value, lc.flux.value, yerr=lc.flux_err.value,
                    fmt='.', color='darkgray', ecolor='lightgrey', markersize=2)
    else:
        ax.plot(lc.time.value, lc.flux.value, '.', color='darkgray', markersize=2)

    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.margins(x=0.0, y=0.1)
    ax.minorticks_on()

    if fit:
        xmin, xmax = ax.get_xlim()
        x = np.linspace(xmin, xmax, 1000)
        plt.plot(x, fit.model(fit.params, x), color='red')

    return ax

def plot_folds(folds, err=True, fit=None, shift_t0=None, figsize=(12,6),
               xlabel='Phase', ylabel='Flux'):
    """ Plot a list of folds.

    Args:
        folds: a list of LightCurve objects
    """
    _, ax = plt.subplots(figsize=figsize)

    for fold in folds:
        ts = fold.time.value
        ys = fold.flux.value
        ys_err = fold.flux_err.value

        if shift_t0:
            ts = ts + shift_t0

        if err:
            ax.errorbar(ts, ys, yerr=ys_err, fmt='.', color='darkgray',
                        ecolor='lightgrey', markersize=2)
        else:
            ax.plot(ts, ys, '.', color='darkgray', markersize=2)

    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.margins(x=0.0, y=0.1)
    ax.minorticks_on()

    if fit:
        x = np.linspace(min(ts), max(ts), 1000)
        plt.plot(x, fit.model(fit.params, x), color='red', linewidth=2)

    return ax
