
from iminuit import Minuit
from iminuit.cost import UnbinnedNLL, BinnedNLL
import numpy as np
import pytest
from emmanoulopoulos.emmanoulopoulos_lc_simulation import power_spectral_density
from scipy.stats import norm, poisson, lognorm, gamma


def test_power_spectral_density():
    from emmanoulopoulos.models import power_spectral_density
    A = 0.0004325697931186756
    alpha_low = 1.7180692792824444
    alpha_high = 6.286876758053707
    f_bend = 0.014238670906664863
    c = 0.0020197057540771723

    f = 1e-3

    p = power_spectral_density(f, A, alpha_low, alpha_high, f_bend, c)

    np.testing.assert_array_almost_equal(p, 61.698685461074916, decimal=20)


def test_create_lc(uneven_times, sample_fluxes_from_bending_PL):
    from emmanoulopoulos.lightcurve import LC

    tbin = 2
    lc_original = LC(time=uneven_times, flux=sample_fluxes_from_bending_PL, errors=0.1*sample_fluxes_from_bending_PL, tbin=tbin)
    
    assert lc_original.interp_length == int((uneven_times.max() - uneven_times.min()) / tbin) + 1
    assert lc_original.interp_length == 205  # results from drawn random number with fixed known seed!


def test_create_lc_bad_length(uneven_times, sample_fluxes_from_bending_PL, tbin):
    from emmanoulopoulos.lightcurve import LC

    with pytest.raises(ValueError) as err:
        lc = LC(time=uneven_times[:-5], flux=sample_fluxes_from_bending_PL, errors=0.1*sample_fluxes_from_bending_PL, tbin=tbin)
    assert err.type is ValueError


def test_f_j(lc):
    f_j = lc.f_j()
    print(f_j)
    assert f_j[0] == 0
    assert len(f_j) == (lc.interp_length - lc.interp_length % 2) / 2 + 1


def test_lc_periodogram(lc):
    f_j, P_j = lc.periodogram()

    assert len(f_j) == len(P_j)
    assert len(f_j) == (lc.interp_length - lc.interp_length % 2) / 2 + 1


def test_fit_PSD(lc, psd_parameter):
    psd_parameter_fit = lc.fit_PSD()

    assert psd_parameter_fit["A"] is not None
    assert psd_parameter_fit["alpha_high"] is not None


def test_unbinned_fit_PDF(lc):
    pdf_unbinnned_fit = lc.fit_PDF(unbinned=True)

    assert pdf_unbinnned_fit.to_dict()["a"] > 0
    assert (pdf_unbinnned_fit.to_dict()["p"] > 0) and (pdf_unbinnned_fit.to_dict()["p"] < 1)


def test_binned_fit_PDF(lc):
    pdf_binnned_fit = lc.fit_PDF(unbinned=False)

    assert pdf_binnned_fit.to_dict()["a"] > 0
    assert (pdf_binnned_fit.to_dict()["p"] > 0) and (pdf_binnned_fit.to_dict()["p"] < 1)
