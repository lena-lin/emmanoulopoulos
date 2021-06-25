import numpy as np
import numpy.random as rnd
from iminuit import Minuit
from iminuit.cost import UnbinnedNLL, BinnedNLL
from emmanoulopoulos.models import power_spectral_density, periodogram_pdf, sample_gamma_lognorm
from emmanoulopoulos.lightcurve import LC

DEFAULT_RNG = np.random.default_rng(5)



def timmer_koenig(lc, psd_model=power_spectral_density, red_noise_factor=1, alias_tbin=1):
    '''
    Simulate lightcurve from given power spectrum as proposed by Timmer & Koenig (https://arxiv.org/pdf/1305.0304.pdf , 1995)
    The resulting lightcurve represents only the given power spectral density (PSD)
    but no specific probability distribution function (PDF) of the flux values.
    TK algorithm:
        - Create Fourier components for every frequency f_j by sampling real and imaginary part from normal distributions (mean=0, std=1).
        - Multiply the components with the given PSD evaluated at the respective f_j: PSD(f_j, parameter)
        - Apply inverse FFT to obtain the simulated lightcurve (this lightcurve may have negative "flux points")
        (-Scale the simulated lightcurve to a desired mean and standard deviation, e.g. the ones from the original lightcurve)
    '''
    psd_parameter = lc.psd_parameter.to_dict().copy()
    t_bin = lc.tbin / alias_tbin

    # set PSD parameter c (containing Poisson noise) to 0
    # (https://arxiv.org/pdf/1305.0304.pdf page 3, section 2.1)
    psd_parameter['c'] = 0

    N_red_noise = red_noise_factor * lc.interp_length * alias_tbin
    j_max = int((N_red_noise - (N_red_noise % 2)) / 2)
    
    freq = 1 / (N_red_noise * lc.tbin / alias_tbin) * np.arange(1, j_max + 1, 1)
    PSD = power_spectral_density(freq, **psd_parameter)

    # sample real and imaginary part from normal distributions
    # TK eq. 9 + page 709
    norm = np.sqrt(PSD / 2)
    real = rnd.normal(0, 1, j_max)
    imag = rnd.normal(0, 1, j_max - 1)
    
    # if N_red_noise is even, the last sample represents the Nyquist frequency and the imaginary part must be 0
    if N_red_noise % 2 == 0:
        imag = np.append(imag, 0)
        complex_numbers = norm * (real + 1j * imag)

    else:
        imag = np.append(imag, rnd.normal(0, 1, 1))
        complex_numbers = norm * (real + 1j * imag)

    complex_numbers = np.append(0, complex_numbers)
    
    # inverse FFT of the sampled components to obtain the lightcurve of lenght N_red_noise
    inverse_fft =  np.fft.irfft(complex_numbers, N_red_noise)
    
    # take subset of the long lightcurve of length N
    if red_noise_factor > 1:
        extract = rnd.randint(lc.interp_length * alias_tbin - 1, lc.interp_length * alias_tbin * (red_noise_factor - 1))
        lightcurve = inverse_fft[extract : extract + lc.interp_length * alias_tbin]
    else:
        lightcurve = inverse_fft
        
    if alias_tbin != 1:
        lightcurve = lightcurve[::alias_tbin]
        
    # scale the flux values to the mean and standard deviation of the original lightcurve to avoid negative flux points
    lightcurve = (lightcurve - np.mean(lightcurve)) / np.std(lightcurve) * lc.interp_flux.std() + lc.interp_flux.mean()
    
    # create new lightcurve from the simulated flux points
    lc_tk = LC(time=lc.interp_time, flux=lightcurve, tbin=lc.tbin)
    
    return lc_tk


class Emmanoulopoulos_Sampler:
    def __init__(self, lc, poisson_noise=True, tk_red_noise_factor=100):
        self.lc = lc
        self.poisson_noise = poisson_noise
        self.tk_red_noise_factor = tk_red_noise_factor
        
    
    def periodogram_from_fft(self, fft_comp, tbin, mean, N):
        return (2 * tbin) / (mean**2 * N) * np.abs(fft_comp)**2

    
    def amplitude(self, fft_comp, N):
        return 1 / N * np.abs(fft_comp)

    
    def fit_PSD(self, f_j, P_j):

        nll = UnbinnedNLL(
            data=[f_j, P_j],
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

        return m.values
        
    
    def simulate_lc(self):
        
        N = self.lc.interp_length
        tbin = self.lc.tbin
        
        # produce Timmer&Koenig lightcurve representing the underlying PSD, but not PDF (step i)
        lc_tk = timmer_koenig(self.lc, red_noise_factor=self.tk_red_noise_factor)
        
        # FFT of real valued lightcurve 
        lc_tk_rfft = np.fft.rfft(lc_tk.original_flux)
        ampl_norm = self.amplitude(lc_tk_rfft, N)
        phase_norm = np.angle(lc_tk_rfft)
        P_j_norm = self.periodogram_from_fft(lc_tk_rfft, self.lc.tbin, lc_tk.original_flux_mean, lc_tk.original_length)
        psd_fit_result = self.fit_PSD(self.lc._f_periodogram, P_j_norm).to_dict()

        # sample white noise data from fitted PDF
        lc_sim = sample_gamma_lognorm(*self.lc.pdf_parameter, size=N)

        assert lc_sim.all() > 0

        converge = False
        n_loop = 0

        while not converge:

            n_loop+=1

            # amplitude, phase and periodogram from white noise data (step ii)
            lc_sim_fft = np.fft.rfft(lc_sim)
            ampl_sim = self.amplitude(lc_sim_fft, N)
            phase_sim = np.angle(lc_sim_fft)
            P_j_sim = self.periodogram_from_fft(lc_sim_fft, tbin, lc_sim.mean(), N)

            # combine ampl_norm and phase_sim to lc_adjust (step iii)
            fft_adjust = ampl_norm * np.exp(1j * phase_sim)
            lc_adjust = np.fft.irfft(fft_adjust, N)

            # replace highest value of lc_adjust with highest values of lc_sim and so on (step iv)
            argsorted_lc_sim = np.argsort(lc_sim)
            argsorted_lc_adjust = np.argsort(lc_adjust)

            lc_adjust[argsorted_lc_adjust] = lc_sim[argsorted_lc_sim]
            lc_sim = lc_adjust

            assert all(lc_sim > 0)
            P_j_sim = self.periodogram_from_fft(np.fft.rfft(lc_sim), tbin, lc_sim.mean(), N)

            psd_fit_result_sim = self.fit_PSD(self.lc._f_periodogram, P_j_sim).to_dict()

            for k,v in psd_fit_result.items():
                if k not in ['alpha_low', 'alpha_high', 'f_bend']:
                    continue
                if abs(psd_fit_result_sim[k] - v) > 0.0001:
                    break
            else:
                print("converge")
                converge = True

            psd_fit_result = psd_fit_result_sim
            
        if self.poisson_noise:

            lc_sim = DEFAULT_RNG.poisson(lc_sim * tbin) / tbin
        
        try:
            assert all(lc_sim >= 0)
        except AssertionError:
            print(lc_sim)
            raise ValueError
            
        flux_original_sampling = np.interp(self.lc.original_time, self.lc.interp_time, lc_sim)
        

        return LC(flux=flux_original_sampling, time=self.lc.original_time, tbin=self.lc.tbin)
    