from emmanoulopoulos.emmanoulopoulos_lc_simulation import Emmanoulopoulos_Sampler
from emmanoulopoulos.lightcurve import LC
from astropy.table import Table


def load_data():
    t_vlba = Table.read('/home/llinhoff/Documents/Thesis/Data/Radio_Components_Flux/flux_components_43_calibrated.fits')

    time = t_vlba['MJD']
    flux = t_vlba['C1']
    err = t_vlba['C1_ERR']

    return time, flux, err


if __name__ == "__main__":
    time, flux, err = load_data()
    lc = LC(time, flux, err, tbin=10)
    lc.fit_PSD()
    lc.fit_PDF()

    Emma = Emmanoulopoulos_Sampler(lc)
    lc_sim = Emma.simulate_lc()

    print(lc_sim.original_flux)
