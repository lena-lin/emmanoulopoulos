import numpy as np


def test_emmanoulopoulos(lc):
    from emmanoulopoulos.emmanoulopoulos_lc_simulation import Emmanoulopoulos_Sampler

    sampler = Emmanoulopoulos_Sampler(lc)

    sim_lc = sampler.simulate_lc()

    fit_sim_lc = sim_lc.fit_PSD().to_dict()
    fit_lc = lc.fit_PSD().to_dict()

    assert sim_lc.original_length == lc.original_length
    np.testing.assert_allclose(sim_lc.interp_flux_mean, lc.interp_flux_mean, rtol=0.2)
    np.testing.assert_allclose(fit_sim_lc["alpha_low"], fit_lc["alpha_low"], rtol=1)
    np.testing.assert_allclose(fit_sim_lc["alpha_high"], fit_lc["alpha_high"], rtol=1)
    np.testing.assert_allclose(fit_sim_lc["f_bend"], fit_lc["f_bend"], rtol=5)

