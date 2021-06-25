import pytest
import numpy as np

def test_create_tk(lc):
    from emmanoulopoulos.emmanoulopoulos_lc_simulation import timmer_koenig

    sim_lc = timmer_koenig(lc)
    fit_sim_lc = sim_lc.fit_PSD().to_dict()
    fit_lc = lc.fit_PSD().to_dict()

    assert sim_lc.original_length == lc.interp_length
    np.testing.assert_almost_equal(sim_lc.interp_flux_mean, lc.interp_flux_mean, decimal=2)
    np.testing.assert_allclose(fit_sim_lc["alpha_low"], fit_lc["alpha_low"], rtol=0.2)
    np.testing.assert_allclose(fit_sim_lc["alpha_high"], fit_lc["alpha_high"], rtol=1)
    np.testing.assert_allclose(fit_sim_lc["f_bend"], fit_lc["f_bend"], rtol=10)


def test_tk_red_noise(lc):
    from emmanoulopoulos.emmanoulopoulos_lc_simulation import timmer_koenig
    sim_lc = timmer_koenig(lc, red_noise_factor=100)
    fit_sim_lc = sim_lc.fit_PSD().to_dict()
    fit_lc = lc.fit_PSD().to_dict()

    assert sim_lc.original_length == lc.interp_length
    np.testing.assert_almost_equal(sim_lc.interp_flux_mean, lc.interp_flux_mean, decimal=2)
    np.testing.assert_allclose(fit_sim_lc["alpha_low"], fit_lc["alpha_low"], rtol=0.5)
    np.testing.assert_allclose(fit_sim_lc["alpha_high"], fit_lc["alpha_high"], rtol=0.5)
    np.testing.assert_allclose(fit_sim_lc["f_bend"], fit_lc["f_bend"], rtol=5)


def test_tk_alias(lc):
    from emmanoulopoulos.emmanoulopoulos_lc_simulation import timmer_koenig
    sim_lc = timmer_koenig(lc, alias_tbin=10)
    fit_sim_lc = sim_lc.fit_PSD().to_dict()
    fit_lc = lc.fit_PSD().to_dict()

    assert sim_lc.original_length == lc.interp_length
    np.testing.assert_almost_equal(sim_lc.interp_flux_mean, lc.interp_flux_mean, decimal=2)
    np.testing.assert_allclose(fit_sim_lc["alpha_low"], fit_lc["alpha_low"], rtol=0.5)
    np.testing.assert_allclose(fit_sim_lc["alpha_high"], fit_lc["alpha_high"], rtol=0.5)
    np.testing.assert_allclose(fit_sim_lc["f_bend"], fit_lc["f_bend"], rtol=5)


def test_tk_alias_red_noise(lc):
    from emmanoulopoulos.emmanoulopoulos_lc_simulation import timmer_koenig
    sim_lc = timmer_koenig(lc, red_noise_factor=100, alias_tbin=10)
    fit_sim_lc = sim_lc.fit_PSD().to_dict()
    fit_lc = lc.fit_PSD().to_dict()

    assert sim_lc.original_length == lc.interp_length
    np.testing.assert_almost_equal(sim_lc.interp_flux_mean, lc.interp_flux_mean, decimal=2)
    np.testing.assert_allclose(fit_sim_lc["alpha_low"], fit_lc["alpha_low"], rtol=0.5)
    np.testing.assert_allclose(fit_sim_lc["alpha_high"], fit_lc["alpha_high"], rtol=0.5)
    np.testing.assert_allclose(fit_sim_lc["f_bend"], fit_lc["f_bend"], rtol=5)