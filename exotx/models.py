
from typing import List
import logging, copy

from .classes import Params, LightCurve
from .plotting import COLORS

from scipy import stats
import numpy as np
from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive
import emcee, batman
import matplotlib.pyplot as plt
from matplotlib import ticker
from tqdm import tqdm

class LightCurveModel():
    params_names = ['t0', 'p', 'rp', 'a', 'i', 'e', 'w', 'ld', 'u']
    params_defaults = {
        't0': 0.0, 'p': 1, 'rp': 0.1, 'a': 1, 'i': 90,
        'e': 0.0, 'w': 90, 'ld': 'quadratic', 'u': [0.3, 0.3]
    }

    def __init__(self, params, free_params = []):
        ps, defaults = Params(), []

        for p in self.params_names:
            value = params.get(p)
            if value is None:
                value = self.params_defaults[p]
                defaults.append(p)
                logging.info(f"Using default value for param { p }: { value }")
            ps.set(p, value)

        self.params = ps
        self.init_params = params
        self.free_params = free_params
        self.defaults = defaults
        self.time = np.array([])
        self.flux = np.array([])
        self.flux_err = np.array([])
        self.priors = {}
        self._initial_params = []

    def fit(self, lc: LightCurve):
        if isinstance(lc, LightCurve):
            self.time = lc.time.value
            self.flux = lc.flux.value
            self.flux_err = lc.flux_err.value

        # setup
        data = (self.time, self.flux, self.flux_err)
        self._init_mcmc()

        # mcmc burnin
        sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self._log_probability, args=data)
        logging.info(f"Run MCMC burn-in, { int(self.steps/2) } steps..")
        self.p0, _, _ = sampler.run_mcmc(self.p0, int(self.steps/2), progress=True)
        sampler.reset()

        # run mcmc
        logging.info(f"Run MCMC, { self.steps } steps..")
        self.p0, _, _ = sampler.run_mcmc(self.p0, self.steps, progress=True)

        # store samples chain
        self.samples = sampler.get_chain()

        # update params
        flat_samples = sampler.get_chain(flat = True)
        for i in range(len(self.free_params)):
            self.params.set(self.free_params[i], np.median(flat_samples[:, i]))

        logging.info(f"MCMC done")

    def _init_mcmc(self):
        logging.info('Init MCMC')

        self._initial_params = []
        self._init_priors_initial_params()

        self.jitter = 0
        self.ndim = len(self.free_params)
        self.nwalkers = 32
        self.steps = 2000
        self.p0 = [np.array(self._initial_params) + 1e-8 * np.random.randn(self.ndim) for i in range(self.nwalkers)]

    def _init_priors_initial_params(self):
        logging.info('Init priors and value for free params')

        if 't0' in self.free_params:
            self.priors['t0'] = stats.norm(loc=self.params.t0, scale=0.001)
            self._initial_params.append(self.params.t0)
        if 'p' in self.free_params:
            self.priors['p'] = stats.norm(loc=self.params.p, scale=0.001)
            self._initial_params.append(self.params.p)
        if 'rp' in self.free_params:
            if 'rp' in self.defaults:
                table = NasaExoplanetArchive.query_criteria(table='exoplanets', select='pl_rads,st_rad', where="pl_rads != 0.0 and st_rad != 0.0")
                data = table['pl_rads']/table['st_rad']
                data = list(filter(lambda x: x < 1, data.to_value()))
                mu, std = stats.norm.fit(data)
                self.priors['rp'] = stats.norm(loc=mu, scale=0.1)
                self._initial_params.append(mu)
            else:
                self.priors['rp'] = stats.norm(loc=self.params.rp, scale=0.1)
                self._initial_params.append(self.params.rp)
        if 'a' in self.free_params:
            if 'a' in self.defaults:
                table = NasaExoplanetArchive.query_criteria(table='exoplanets', select='pl_orbsmax,st_rad', where="pl_orbsmax != 0.0 and st_rad != 0.0")
                data = table['pl_orbsmax']/table['st_rad']
                mu, std = stats.norm.fit(data.to_value())
                self.priors['a'] = stats.norm(loc=mu, scale=std)
                self._initial_params.append(mu)
            else:
                self.priors['a'] = stats.norm(loc=self.params.a, scale=0.1)
                self._initial_params.append(self.params.a)
        if 'i' in self.free_params:
            if 'i' in self.defaults:
                data = NasaExoplanetArchive.query_criteria(table='exoplanets', select='pl_orbincl', where="pl_orbincl != 0.0 and pl_discmethod = 'Transit'")['pl_orbincl']
                mu, std = stats.norm.fit(data.to_value())
                self.priors['i'] = stats.norm(loc=mu, scale=std)
                self._initial_params.append(mu)
            else:
                self.priors['i'] = stats.norm(loc=self.params.i, scale=4.0)
                self._initial_params.append(self.params.i)

    def _log_probability(self, params, t, y, yerr):
        lp = self._log_prior(params)

        if not np.isfinite(lp):
            return -np.inf

        return lp + self._log_likelihood(params, t, y, yerr)

    def _log_prior(self, params):
        total_p = 0

        for i in range(len(params)):
            p = self.priors[self.free_params[i]].logpdf(params[i])
            if not np.isinf(p):
                total_p += p

        return total_p

    def _log_likelihood(self, params, t, y, yerr):
        inv_sigma2 = 1.0/(yerr**2 + self._model_batman(params, t) **2*np.exp(2*self.jitter))

        return -0.5*(np.sum( (y - self._model_batman(params, t)) **2 *inv_sigma2 - np.log(inv_sigma2)))

    def _model_batman(self, params, t):
        params_bat = batman.TransitParams()

        if 't0' in self.free_params:
            params_bat.t0 = params[self.free_params.index('t0')]
        else:
            params_bat.t0 = self.params.t0
        if 'p' in self.free_params:
            params_bat.per = params[self.free_params.index('p')]
        else:
            params_bat.per = self.params.p
        if 'rp' in self.free_params:
            params_bat.rp = params[self.free_params.index('rp')]
        else:
            params_bat.rp = self.params.rp
        if 'a' in self.free_params:
            params_bat.a = params[self.free_params.index('a')]
        else:
            params_bat.a = self.params.a
        if 'i' in self.free_params:
            params_bat.inc = params[self.free_params.index('i')]
        else:
            params_bat.inc = self.params.i
        if 'e' in self.free_params:
            params_bat.ecc = params[self.free_params.index('e')]
        else:
            params_bat.ecc = self.params.e
        if 'w' in self.free_params:
            params_bat.w = params[self.free_params.index('w')]
        else:
            params_bat.w = self.params.w
        if 'ld' in self.free_params:
            params_bat.limb_dark = params[self.free_params.index('ld')]
        else:
            params_bat.limb_dark = self.params.ld
        if 'u' in self.free_params:
            params_bat.u = params[self.free_params.index('u')]
        else:
            params_bat.u = self.params.u

        m = batman.TransitModel(params_bat, t)

        return m.light_curve(params_bat)

    def summary(self):
        _str = f"Init params: { self.init_params }\nModel params: { self.params }\nFree params: { self.free_params }\n\n"

        print(_str)

    def model(self, t, params=None):
        params_bat = batman.TransitParams()

        if params is None:
            params = self.params

        params_bat.t0 = params.t0
        params_bat.per = params.p
        params_bat.rp = params.rp
        params_bat.a = params.a
        params_bat.inc = params.i
        params_bat.ecc = params.e
        params_bat.w = params.w
        params_bat.limb_dark = params.ld
        params_bat.u = params.u

        m = batman.TransitModel(params_bat, t)

        return m.light_curve(params_bat)

    def plot(self, err=False, figsize=(12,4), xlabel='Phase', ylabel='Flux ($e^- s^{-1}$)'):
        _, ax = plt.subplots(figsize=figsize)

        if err:
            ax.errorbar(self.time, self.flux, yerr=self.flux_err, fmt='.', color=COLORS[0],
                        ecolor=COLORS[1], markersize=2)
        else:
            ax.plot(self.time, self.flux, '.', color=COLORS[0], markersize=2)

        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-1,1))

        ax.set_xlabel(xlabel, fontsize=16)
        ax.set_ylabel(ylabel, fontsize=16)
        ax.margins(x=0.0, y=0.1)
        ax.minorticks_on()
        ax.yaxis.set_major_formatter(formatter)

        if len(self.time) > 0:
            _min, _max = min(self.time), max(self.time)
        else:
            _min, _max = ax.get_xlim()
        x = np.linspace(_min, _max, 1000)
        ax.plot(x, self.model(x), color='red', linewidth=2)

        return ax

    def plot_residuals(self, figsize=(12,4)):
        _, ax = plt.subplots(figsize=figsize)

        final = self.model(self.time)

        logging.info('Computing residuals for all walkers')
        for walker in tqdm(range(self.nwalkers)):
            norms = np.zeros(self.steps)
            for step in range(self.steps):
                ps = copy.deepcopy(self.params)
                for i in range(len(self.free_params)):
                    ps.set(self.free_params[i], self.samples[step, walker, i])
                curr = self.model(self.time, params=ps)
                norms[step] = np.linalg.norm(final - curr)
            ax.plot(list(range(self.steps)), norms)

        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-1,1))

        ax.set_xlabel('Step', fontsize=16)
        ax.set_ylabel('Residual', fontsize=16)
        ax.margins(x=0.0, y=0.1)
        ax.minorticks_on()
        ax.yaxis.set_major_formatter(formatter)

        return ax
