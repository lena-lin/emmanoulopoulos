import numpy as np
from scipy.stats import gamma, lognorm

DEFAULT_RNG = np.random.default_rng(4)


def power_spectral_density(f, A, alpha_low, alpha_high, f_bend, c):
    '''
    Bending Power Law as used in
    Eq. (2) of https://arxiv.org/pdf/1305.0304.pdf
    '''
    num = A * f**(-alpha_low)
    denom = 1 + (f / f_bend)**(alpha_high - alpha_low)
    return num / denom + c


def periodogram_pdf(x, A, alpha_low, alpha_high, f_bend, c):
    '''
    PDF for likelihood-fitting the PSD, according to 
    https://arxiv.org/pdf/1305.0304.pdf , page 21, eq. (A8)
    '''
    f, p = x
    f = f[1:]
    p = p[1:]
    psd = power_spectral_density(f, A, alpha_low, alpha_high, f_bend, c)

    # N even
    if len(f) % 2 == 0:
        pdf = np.empty(len(f))
        pdf[:-1] = gamma.pdf(p[:-1], a=1, scale=psd[:-1])
        pdf[-1] = gamma.pdf(p[-1], a=0.5, scale=psd[-1])
        return pdf

    # N odd
    return 0.5 * gamma.pdf(p, a=1, scale=psd)


def cdf_gamma_lognorm(x, a, s, loc, scale, p):
    '''
    Model for PDF as used in https://arxiv.org/pdf/1305.0304.pdf , page 7 eq. (3)
    '''
    return p * gamma.cdf(x, a) + (1 - p) * lognorm.cdf(x, s, loc, scale)


def pdf_gamma_lognorm(x, a, s, loc, scale, p):
    return p * gamma.pdf(x, a) + (1 - p) * lognorm.pdf(x, s, loc, scale)


def sample_gamma_lognorm(a, s, loc, scale, p, size, rng=None,):
    rng = rng or DEFAULT_RNG

    size1 = rng.poisson(p * size)
    size2 = size - size1
    
    return np.append(gamma.rvs(a=a, size=size1, random_state=rng), lognorm.rvs(s=s, loc=loc, scale=scale, size=size2, random_state=rng))