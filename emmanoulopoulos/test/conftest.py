import numpy as np
import pytest
from emmanoulopoulos.lightcurve import LC
from emmanoulopoulos.models import power_spectral_density

def PL(v, A, beta):
    p = A * np.power(v, -beta)
    return p


@pytest.fixture(scope='session')
def tbin():
    return 2


@pytest.fixture(scope='session')
def psd_parameter():
    return {"A": 0.02, "alpha_low": 1, "alpha_high": 5, "f_bend": 0.01, "c": 0}



@pytest.fixture(scope='session')
def uneven_times():
    tm = np.array(sorted([a + np.random.default_rng(42).poisson(a, 1)[0] for a in range(0,200)]))
    mask = (np.diff(tm) != 0)
    t = tm[1:][mask]
    return t


@pytest.fixture(scope='session')
def sample_fluxes_from_bending_PL(uneven_times, psd_parameter):
    N=len(uneven_times)

    j_max = int((N - (N % 2)) / 2)
    tbin=1

    freq = 1 / (N * tbin) * np.arange(1, j_max + 1, 1)
    PSD = power_spectral_density(freq, **psd_parameter)

    # sample real and imaginary part from normal distributions
    # TK eq. 9 + page 709
    norm = np.sqrt(PSD / 2)
    real = np.random.default_rng(42).normal(0, 1, j_max)
    imag = np.random.default_rng(42).normal(0, 1, j_max - 1)

    # if N_red_noise is even, the last sample represents the Nyquist frequency and the imaginary part must be 0
    if N % 2 == 0:
        imag = np.append(imag, 0)
        complex_numbers = norm * (real + 1j * imag)

    else:
        imag = np.append(imag, np.random.default_rng(42).normal(0, 1, 1))
        complex_numbers = norm * (real + 1j * imag)

    complex_numbers = np.append(0, complex_numbers)

    # inverse FFT of the sampled components to obtain the lightcurve of lenght N_red_noise
    inverse_fft =  np.fft.irfft(complex_numbers, N)
    inverse_fft = inverse_fft - inverse_fft.min() + 10

    return inverse_fft


@pytest.fixture(scope='session')
def lc(uneven_times, sample_fluxes_from_bending_PL, tbin):

    return LC(time=uneven_times, flux=sample_fluxes_from_bending_PL, errors=0.1*sample_fluxes_from_bending_PL, tbin=tbin)
