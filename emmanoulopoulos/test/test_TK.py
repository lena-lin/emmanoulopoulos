import pytest
import numpy as np

def test_create_tk(lc, TK):

    sim_lc = TK.sample_from_lc(lc)
    fit_sim_lc = sim_lc.fit_PSD().to_dict()
    fit_lc = lc.fit_PSD().to_dict()

    assert sim_lc.original_length == lc.interp_length
    np.testing.assert_almost_equal(sim_lc.interp_flux_mean, lc.interp_flux_mean, decimal=2)
    np.testing.assert_allclose(fit_sim_lc["alpha_low"], fit_lc["alpha_low"], rtol=0.2)
    np.testing.assert_allclose(fit_sim_lc["alpha_high"], fit_lc["alpha_high"], rtol=1)
    np.testing.assert_allclose(fit_sim_lc["f_bend"], fit_lc["f_bend"], rtol=10)


def test_tk_red_noise(lc):
    from emmanoulopoulos.emmanoulopoulos_lc_simulation import TimmerKoenig
    TK = TimmerKoenig(red_noise_factor=100)
    sim_lc = TK.sample_from_lc(lc)
    fit_sim_lc = sim_lc.fit_PSD().to_dict()
    fit_lc = lc.fit_PSD().to_dict()

    assert sim_lc.original_length == lc.interp_length
    np.testing.assert_almost_equal(sim_lc.interp_flux_mean, lc.interp_flux_mean, decimal=2)
    np.testing.assert_allclose(fit_sim_lc["alpha_low"], fit_lc["alpha_low"], rtol=0.5)
    np.testing.assert_allclose(fit_sim_lc["alpha_high"], fit_lc["alpha_high"], rtol=0.5)
    np.testing.assert_allclose(fit_sim_lc["f_bend"], fit_lc["f_bend"], rtol=5)


def test_tk_alias(lc):
    from emmanoulopoulos.emmanoulopoulos_lc_simulation import TimmerKoenig
    TK = TimmerKoenig(alias_tbin=10)
    sim_lc = TK.sample_from_lc(lc)
    fit_sim_lc = sim_lc.fit_PSD().to_dict()
    fit_lc = lc.fit_PSD().to_dict()

    assert sim_lc.original_length == lc.interp_length
    np.testing.assert_almost_equal(sim_lc.interp_flux_mean, lc.interp_flux_mean, decimal=2)
    np.testing.assert_allclose(fit_sim_lc["alpha_low"], fit_lc["alpha_low"], rtol=0.5)
    np.testing.assert_allclose(fit_sim_lc["alpha_high"], fit_lc["alpha_high"], rtol=0.5)
    np.testing.assert_allclose(fit_sim_lc["f_bend"], fit_lc["f_bend"], rtol=5)


def test_tk_alias_red_noise(lc):
    from emmanoulopoulos.emmanoulopoulos_lc_simulation import TimmerKoenig
    TK = TimmerKoenig(red_noise_factor=100, alias_tbin=10)
    sim_lc = TK.sample_from_lc(lc)
    fit_sim_lc = sim_lc.fit_PSD().to_dict()
    fit_lc = lc.fit_PSD().to_dict()

    assert sim_lc.original_length == lc.interp_length
    np.testing.assert_almost_equal(sim_lc.interp_flux_mean, lc.interp_flux_mean, decimal=2)
    np.testing.assert_allclose(fit_sim_lc["alpha_low"], fit_lc["alpha_low"], rtol=0.5)
    np.testing.assert_allclose(fit_sim_lc["alpha_high"], fit_lc["alpha_high"], rtol=0.5)
    np.testing.assert_allclose(fit_sim_lc["f_bend"], fit_lc["f_bend"], rtol=5)


def test_tk_sample_from_psd(TK):
    from emmanoulopoulos.emmanoulopoulos_lc_simulation import TimmerKoenig
    TK = TimmerKoenig(red_noise_factor=1000, alias_tbin=10)

    psd_params = {"A": 0.02, "alpha_low": 1, "alpha_high": 5, "f_bend": 0.01, "c": 0}

    lc_tk = TK.sample_from_psd(psd_params, tbin=1, N=1000)

    fit_psd = lc_tk.fit_PSD()

    assert lc_tk.original_length == 1000

    # np.testing.assert_allclose(fit_psd["alpha_low"], psd_params["alpha_low"], rtol=2)
    np.testing.assert_allclose(fit_psd["alpha_high"], psd_params["alpha_high"], rtol=0.2)
    # np.testing.assert_almost_equal(fit_psd["f_bend"], psd_params["f_bend"], decimal=2)
    # np.testing.assert_allclose(fit_psd["A"], psd_params["A"], rtol=0.1)
    # np.testing.assert_allclose(fit_psd["c"], psd_params["c"], atol=0.1)