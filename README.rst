
Exoplanets Transit Explorer
===========================

.. warning:: This package is experimental, and under heavy development.

An experimental package for quickly exploring exoplanets transits and associated resources.

Quick Start
===========

Install package from the git repository:

.. code-block:: bash

    $ pip install git+https://github.com/nunorc/exotx@master

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
    lc =  exotx.combine(lcs)

    # plot the final result
    exotx.plot_lc(lc)

Acknowledgments
===============

Thank you to the authors of the
`lightkurve <https://docs.lightkurve.org>`_ and related packages.

Thank you to Susana Barros and Olivier Demangeon for the comments
and discussions that helped improve this package.
