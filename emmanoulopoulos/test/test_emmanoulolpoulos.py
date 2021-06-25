import numpy as np


def test_emmanoulopoulos():
    from emmanoulopoulos.emmanoulopoulos_lc_simulation import Emmanoulopoulos_Sampler

    N = 1000
    tbin = 1

    psd_params = {"A": 0.02, "alpha_low": 1, "alpha_high": 5, "f_bend": 0.01, "c": 0}
    pdf_params = {"a": 1, "s": 1, "loc": 0.5, "scale": 5, "p": 0.6}

    sampler = Emmanoulopoulos_Sampler()

    sim_lc = sampler.sample_from_psd_pdf(psd_params, pdf_params, N, tbin)

    psd_fit_sim_lc = sim_lc.fit_PSD().to_dict()
    pdf_fit_sim_lc = sim_lc.fit_PDF().to_dict()


    assert sim_lc.original_length == N

    assert psd_fit_sim_lc == psd_params
    assert pdf_fit_sim_lc == pdf_params

    # np.testing.assert_allclose(sim_lc.interp_flux_mean, lc.interp_flux_mean, rtol=0.2)
    # np.testing.assert_allclose(psd_fit_sim_lc["alpha_low"], psd_fit_lc["alpha_low"], rtol=1)
    # np.testing.assert_allclose(psd_fit_sim_lc["alpha_high"], psd_fit_lc["alpha_high"], rtol=1)
    # np.testing.assert_allclose(psd_fit_sim_lc["f_bend"], psd_fit_lc["f_bend"], rtol=5)

