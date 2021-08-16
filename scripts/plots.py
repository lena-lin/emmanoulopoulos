import astropy.units as u
from emmanoulopoulos.emmanoulopoulos_lc_simulation import TimmerKoenig
from emmanoulopoulos.emmanoulopoulos_lc_simulation import Emmanoulopoulos_Sampler
from emmanoulopoulos.models import power_spectral_density, pdf_gamma_lognorm
import matplotlib.pyplot as plt



def plot_tk_sample_from_psd():
    TK = TimmerKoenig(red_noise_factor=10, alias_tbin=1)
    psd_params = {"A": 0.02, "alpha_low": 1, "alpha_high": 5, "f_bend": 0.01, "c": 0}
    lc_tk = TK.sample_from_psd(psd_params, tbin=1*u.day, N=1000)
    
    fit_psd = lc_tk.fit_PSD(window=True)
   
    f = lc_tk._f_periodogram

    fig, ax = plt.subplots(constrained_layout=True)
    ax.plot(lc_tk._f_periodogram, lc_tk._periodogram, 'o')
    ax.plot(lc_tk._f_periodogram, power_spectral_density(lc_tk._f_periodogram, **psd_params), label='original PSD')
    ax.plot(lc_tk._f_periodogram, power_spectral_density(lc_tk._f_periodogram, **fit_psd.to_dict()), label='fitted PSD, window')
    ax.loglog()
    ax.legend()
    plt.savefig("build/tk_from_PSD.pdf")


def plot_emma_psd_pdf():

    N = 100 
    tbin = 30 * u.day

    psd_params = {"A": 0.00008, "alpha_low": 2, "alpha_high": 15, "f_bend": 0.02, "c": 0}
    pdf_params = {"a": 3.5, "s": 0.7, "loc": 2, "scale": 1, "p": 0.6}

    sampler = Emmanoulopoulos_Sampler(poisson_noise=False, tk_red_noise_factor=1000, tk_alias_tbin=1)
    print("A")

    sim_lc = sampler.sample_from_psd_pdf(psd_params, pdf_params, N, tbin)

    print("B")

    psd_fit_sim_lc = sim_lc.fit_PSD(window=True).to_dict()
    pdf_fit_sim_lc = sim_lc.fit_PDF(unbinned=False).to_dict()

    fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(10,3))
    ax1.plot(sim_lc._f_periodogram, sim_lc._periodogram, 'o', )
    ax1.plot(sim_lc._f_periodogram, power_spectral_density(sim_lc._f_periodogram, **psd_params), label='original PSD')
    ax1.plot(sim_lc._f_periodogram, power_spectral_density(sim_lc._f_periodogram, **psd_fit_sim_lc), label='fitted PSD, window')
    ax1.loglog()
    ax1.legend()

    hist, edges, _ = ax2.hist(sim_lc.original_flux, bins=20, density=True)
    c = (edges[:-1] + edges[1:]) / 2
    ax2.plot(c, pdf_gamma_lognorm(c, **pdf_params), label="original PDF")
    ax2.plot(c, pdf_gamma_lognorm(c, **pdf_fit_sim_lc), label="fitted PDF")
    ax2.legend()
    plt.savefig("build/emma_from_model.pdf")


if __name__ == "__main__":
    plot_tk_sample_from_psd()
    print("emma")
    plot_emma_psd_pdf()