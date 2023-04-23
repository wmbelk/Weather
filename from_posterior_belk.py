# -*- coding: utf-8 -*-
"""
Created on Sun May 31 18:57:57 2020

@author: willi
Adapted to add min and max, from https://docs.pymc.io/notebooks/updating_priors.html
"""
import numpy as np
import scipy as sp
import pymc as pm

# define an emperical distribution


def from_posterior(param, samples, lowest_range=None, highest_range=None, dims=None):
    '''
    Parameters
    ----------
    param : string
            name of the parameter for wich the trace is being passed (as posterior) to create a prior
    samples :   need to see if inferance data will work; 
                trace samples taken in pymc3 running of MCMC
    lowest_range : float
        limit on the lowest the tails will be extended out. default is currently 3 * width
     highest_range : float
        limit on highest the tails will be extended out. default is currently 3 * width

    Returns
    -------
    TensorVariable: pymc interpolated distribution
        pm.distributions.Interpolated(param, x, y)
    param= parameter name
    x = domain 
    y = pdf values from samples

        '''

    smin, smax = np.min(samples), np.max(samples)
    width = smax - smin
    if lowest_range is None:
        lowest_range = smin - 3 * width
    else:
        smin = max(lowest_range, smin)

    if highest_range is None:
        highest_range = smax + 3 * width
    else:
        smax = min(highest_range, smax)
    # print(smin,lowest_range,smax,highest_range)
    # X must be array of monotonically increasing numbers:
    x = np.linspace(smin+width/101, smax-width/101, 100)
    y = sp.stats.gaussian_kde(samples.data)(x)

    # what was never sampled should have a small probability but not 0,
    # so we'll extend the domain and use linear approximation of density on it
    # from 3 times to 1.5 times the full width
    x = np.concatenate([[max(lowest_range, x[0] - 3 * width)],
                        x, [min(x[-1] + 3 * width, highest_range)]])
    # print(x)
    tiny_pdf_val = min(y) * 1e-10
    y = np.concatenate([[tiny_pdf_val], y, [tiny_pdf_val]])
    # samples.unstack().dims
    # assuming it was stacked
    if dims is not None:
        print("dims is not None")
        non_mcmc_coords = dims
        return pm.distributions.Interpolated(f'{param}_post2prior', x, y)#, dims=non_mcmc_coords)
    else:  # infer from samples
        non_mcmc_coords = (x for x 
                           in samples.unstack().dims 
                           if x != 'chain' and x != 'draw')
        if len(list(non_mcmc_coords)) == 0:
            return pm.distributions.Interpolated(f'{param}_post2prior', x, y)
        else:
            return pm.distributions.Interpolated(f'{param}_post2prior', x, y)#, dims=non_mcmc_coords)
