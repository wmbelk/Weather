# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 18:29:15 2021

@author: willi
"""
import pymc as pm


def hierarchial_normal(name, mu=0., sigma=None, dims=None, part_pool_sigma=None, prior_sigma=None, dims_sigma=None, sigma_index=None):
    '''
        lifted core from: https://austinrochford.com/posts/2017-07-09-mrpymc3.html
        For use in Pymc3 models
                created a noncentered Normal distribution with passable priors on
                mu (default=0) & sigma (default = HalCauchy(5))
                note the intent of non-centered means there is a Δ~N(0,1, dims=dims)
    I added pass for dims and sigma_dims; sigma index would be used to broadcast shorter sigma to larger offset dim
        Parameters
        ----------
        name : distributions name
                DESCRIPTION.
        mu : tensorVar or float, optional
                can pass tensor or constant for use in returned Deterministic. The default is 0.
        sigma : float, optional
                value used as sd in HalfNormal dist of sigma_var; Note: if not specified the sd is pm.HalfCauchy(f'σ_{name}', 5.). The default is None.
        dims : tupple of Coords names (dims), optional
                use coordinates defined in the pymc3 model. The default is None.
                                is applied to the Δ term and the returned deterministic
        part_pool_sigma : tensorVar Heirarchial distribustion, optional
                 Hyperperameter used as sd in HalfNormal dist of sigma. The default is None.
        prior_sigma: tensorVar Distribution (could be Interpolated from_posterior_belk)
                        superseedes other sigma inputs and shoves prior into the returned deterministic
        dims_sigma : TYPE, optional
                dims applied to the HalfNormal of sigma, if prior_sigma is not used {then prior_sigma already has its own shape or Dims}. The default is None.
        sigma_index : array, optional
                indexing to allign partial pooled sigma with dim of offset, Δ . The default is None.

        Returns
        -------
        pm.Deterministic(name, μ + Δ * σ[sigma_index], dims=dims)

        '''
#     if mu != 0.:
#         pass # no need for deterministic
# # 		try:
# # 			pass
# #             # dims_mu = mu.dims
# #             # μ = pm.Deterministic(f'μ_{name}', mu)#, dims=dims_mu)
# #         except:
# #             μ = mu
#     else:
    # Mean label/pointer to dist. or const value passed through
    μ = mu

    # if dims == None: #TODO: Verify removing this if doesn't break program.. don't think I need this; If doms=None, pymc3 should be able to handel that
    #     Δ = pm.Normal(f'Δ_{name}', 0., 1.)
    # else:
    Δ = pm.Normal(f'Δ_{name}', 0., 1., dims=dims)#, testval=(.001))

    # Sigma
    if prior_sigma != None:
        # pm.Deterministic(f'σ_{name}', prior_sigma)  # , dims=dims ??
        σ = prior_sigma
    else:
        if part_pool_sigma == None:
            if sigma == None:
                σ = pm.HalfCauchy(f'σ_{name}', beta=5.,
                                  initval=(1.))
            else:
                # half-normal is easier to relate to if specifying a prior
                # HalfCauchy(f'σ_{name}', beta=sigma, dims=dims_sigma)
                σ = pm.HalfNormal(f'σ_{name}', sigma=sigma, dims=dims_sigma,
                                  initval=(1.))
        else:  # use RV for partially pooled establishment
            # HalfCauchy(f'σ_{name}', beta=part_pool_sigma, dims=dims_sigma)
            σ = pm.HalfNormal(f'σ_{name}', sigma=part_pool_sigma, dims=dims_sigma,
                              initval=(1.))

    if dims == None:
        return pm.Deterministic(name, μ + Δ * σ)
    else:
        if dims == dims_sigma:
            return pm.Deterministic(name, μ + Δ * σ, dims=dims)
        else:
            return pm.Deterministic(name, μ + Δ * σ[sigma_index], dims=dims)
