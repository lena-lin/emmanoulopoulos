import astropy.units as u
import numpy.fft as ft
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from scipy.signal.windows import hann
from scipy.stats import norm, poisson, lognorm, gamma
from iminuit import Minuit
from iminuit.cost import UnbinnedNLL, BinnedNLL
import numpy.random as rnd
from emmanoulopoulos.models import (
    periodogram_pdf,
    cdf_gamma_lognorm,
    pdf_gamma_lognorm,
)

class LC:
    @u.quantity_input(original_time=[u.day, u.s], tbin=[u.day, u.s])
    def __init__(self, original_time, original_flux, errors=None, tbin=None):
        if (len(original_time) != len(original_flux)):
            raise ValueError(f"time, flux and errors must have the same length! time: {len(original_time)}, flux: {len(original_flux)}")
        
        self.original_time = original_time
        self.original_flux = original_flux
        self.tbin = tbin

        self.errors = errors

        self._f_periodogram = None
        self._periodogram = None
        
        self.psd_parameter = None
        self.psd_parameter_error = None
        
        self.pdf_parameter = None


    @property
    def tbin(self):
        return self._tbin


    @tbin.setter
    def tbin(self, value):
        if (value is not None) and (value.unit != self.original_time.unit):
            raise ValueError(f"Units not matching! original_time: {self.original_time.unit}, tbin: {value.unit}")
        self._tbin = value


    @property
    def interp_time(self):
        if self.tbin is None:
            self._tbin = int(np.diff(self.original_time).mean())

        return np.arange(self.original_time.value.min(), self.original_time.value.max() + self.tbin.value, self.tbin.value) * self.original_time.unit


    @property
    def interp_flux(self):
        '''
        Scale (to 10) and interpolate flux
        '''
        scaled_flux = self.original_flux / self.original_flux.max() * 10
        return np.interp(self.interp_time.value, self.original_time.value, scaled_flux)


    @property
    def original_length(self):
        return len(self.original_time)


    @property
    def interp_length(self):
        return len(self.interp_time)


    @property
    def interp_flux_mean(self):
        return np.mean(self.interp_flux)


    @property
    def original_flux_mean(self):
        return np.mean(self.original_flux)
        

    def fft(self, flux_values=None):
        if flux_values is None:
            return ft.fft(self.interp_flux)
        else:
            return ft.fft(flux_values)


    def f_j(self):
        if self._f_periodogram is None:
            j_max = int((self.interp_length - (self.interp_length % 2)) / 2)
            f_j = 1 / (self.interp_length * self.tbin) * np.arange(0, j_max + 1, 1)
            self._f_periodogram = f_j
        
        return self._f_periodogram

    @property
    def j_max(self):
        return int((self.interp_length - (self.interp_length % 2)) / 2)


    def periodogram(self, window=False):
        if window:
            hann_window = hann(self.interp_length)
            fft = self.fft(hann_window * self.interp_flux)
        else:
            fft = self.fft()
        rms_norm = (2 * self.tbin) / (self.interp_flux_mean**2 * self.interp_length)
        P_j = rms_norm * (fft.real[:self.j_max + 1]**2 + fft.imag[:self.j_max + 1]**2)
        f_j = self.f_j()
        self._periodogram = P_j

        return self._f_periodogram, self._periodogram

    
    def fit_PSD(self, window=True):
        self.periodogram(window=window)
        
        nll = UnbinnedNLL(
            data=[self._f_periodogram[1:], self._periodogram[1:]],
            pdf=periodogram_pdf
        )

        eps = np.finfo(np.float64).eps

        m = Minuit(nll,  A=1e-3, f_bend=5e-3, alpha_low=1.5, alpha_high=4.5, c=0)
        m.limits['A'] = (eps, None)
        m.limits['f_bend'] = (eps, None)
        m.limits['alpha_low'] = (1, None)
        m.limits['alpha_high'] = (1, None)
        m.limits['c'] = (0, None)
        m.migrad()
        self.psd_parameter = m.values
        
        return m.values
    
    
    def fit_PDF(self, unbinned=True):
        eps = np.finfo(np.float64).eps
        if unbinned:
            nll = UnbinnedNLL(self.interp_flux, pdf_gamma_lognorm)

            minimize_unbinned = Minuit(nll,  a=0.1, s=1, loc=0, scale=1, p=0.5)
            minimize_unbinned.limits['loc'] = (0, None)
            minimize_unbinned.limits['a'] = (eps, None)
            minimize_unbinned.limits['s'] = (eps, None)
            minimize_unbinned.limits['p'] = (0, 1)
            minimize_unbinned.migrad()
            self.pdf_parameter = minimize_unbinned.values
            return minimize_unbinned.values
        
        else:
            hist, edges = np.histogram(
                self.interp_flux,
                bins=50,
                range=[self.interp_flux.min(), self.interp_flux.max()])
            width = np.diff(edges)[0]

            nll = BinnedNLL(hist, edges, cdf_gamma_lognorm)

            minimize_binned = Minuit(nll, a=0.1, s=1, loc=0, scale=1, p=0.5)
            minimize_binned.limits['loc'] = (0, None)
            minimize_binned.limits['a'] = (eps, None)
            minimize_binned.limits['s'] = (eps, None)
            minimize_binned.limits['p'] = (0, 1)
            minimize_binned.migrad()
            self.pdf_parameter = minimize_binned.values
            return minimize_binned.values