
Exoplanets Transit Explorer
===========================

.. warning:: This package is experimental, and under development.

An experimental package for quickly exploring exoplanets transits and associated resources.

Quick Start
===========

Installation
------------

Install package from the git repository:

.. code-block:: bash

    $ pip install git+https://github.com/nunorc/exotx@master


Using light curves
------------------

Retrieve and process some light curves.

.. code-block:: python

    import exotx

    # define the arguments for the object of interest
    args = exotx.Args(target='kepler-210', quarters=[1, 2], cadence='long', pdc=True)

    # retrieve the associated light curves
    lcs = exotx.retrieve(args)

    # detrend and normalize all light curves
    lcs = [exotx.detrend(i) for i in lcs]
    lcs = [exotx.normalize(i) for i in lcs]

    # combine into a single light curve
    lc = exotx.combine(lcs)

    # plot the final result
    exotx.plot_lc(lc)

Yielding the following plot.

.. image:: https://nunorc.github.io/exotx/html/_static/lc_plot_1.png

Folding Light Curves
--------------------

Folding a light curve:

.. code-block:: python

    # define parameteres for a known planet
    params = exotx.Params(p=2.4532, t0=134.092)

    # create a list of folds
    folds = exotx.fold(lc, params)

    # plot the folded light curve
    exotx.plot_folds(folds)

Yielding the following plot.

.. image:: https://nunorc.github.io/exotx/html/_static/lc_plot_2.png


Acknowledgments
===============

Thank you to the authors of the
`lightkurve <https://docs.lightkurve.org>`_,
`batman-package <https://lweb.cfa.harvard.edu/~lkreidberg/batman/>`_ ,
`emcee <https://emcee.readthedocs.io/>`_ ,
and upstream packages.

Thank you to Susana Barros and Olivier Demangeon for the discussions that
helped improve this package.
