from emmanoulopoulos.emmanoulopoulos_lc_simulation import Emmanoulopoulos_Sampler
from emmanoulopoulos.lightcurve import LC
from astropy.table import Table
from pathlib import Path
import astropy.units as u
import json
from json import encoder




def load_data():
    t_vlba = Table.read('/home/llinhoff/Documents/Thesis/Data/Radio_Components_Flux/flux_components_43_calibrated.fits')

    mjd = t_vlba['MJD']
    time = (mjd - mjd.min()) * u.day
    flux = t_vlba['C1']
    err = t_vlba['C1_ERR']

    return time, flux, err


if __name__ == "__main__":
    output_path = "/home/llinhoff/Documents/Thesis/Data/LC_Correlation/LC_Sim"
    n = 3
    name = "VLBA_43_C1_calibrated"

    encoder.FLOAT_REPR = lambda o: format(o, '.2f')

    time, flux, err = load_data()
    lc = LC(time, flux, err, tbin=10 * u.day)
    lc.fit_PSD()
    lc.fit_PDF()

    print(lc.pdf_parameter)
    simulated_lcs = {}
    Emma = Emmanoulopoulos_Sampler()
    for i in range(n):
        print(i)
        lc_sim = Emma.sample_from_lc(lc)
        simulated_lcs[i] = list(lc_sim.original_flux)

    with open(str(Path(output_path, "sim_lc_{}_n{}.json".format(name, n))), 'w') as f:
        json.dump(simulated_lcs, f,  indent=4, separators=(", ", ": "), sort_keys=True)

