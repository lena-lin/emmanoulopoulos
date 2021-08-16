from astropy.table import Table
import astropy.units as u
from emmanoulopoulos.emmanoulopoulos_lc_simulation import Emmanoulopoulos_Sampler
from emmanoulopoulos.lightcurve import LC
import json
from json import encoder
import logging
from pathlib import Path
from tqdm import tqdm


logger = logging.getLogger(__name__)
logging.getLogger(__name__).setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s:%(levelname)s:%(message)s", datefmt="%Y-%m-%dT%H:%M:%s"
)
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logging.getLogger().addHandler(handler)


def run_lc_simulation(time, flux, err, name, output_path, n=10,):
    
    lc = LC(time, flux, err, tbin=10 * u.day)
    lc.fit_PSD()
    pdf_parameter = lc.fit_PDF()

    pdf_positive = False
    
    while not pdf_positive:
        for k, v in pdf_parameter.to_dict().items():
            if v < 0:
                pdf_parameter = lc.fit_PDF()
                break
        else:
            pdf_positive = True

    logger.info(f"PDF parameters: {pdf_parameter.to_dict()}")
    logger.info(f"PSD parameters: {lc.psd_parameter.to_dict()}")

    simulated_lcs = {}
    Emma = Emmanoulopoulos_Sampler()
    for i in tqdm(range(n)):
        lc_sim = Emma.sample_from_lc(lc)
        simulated_lcs[i] = list(lc_sim.original_flux)

    with open(str(Path(output_path, "sim_lc_{}_n{}.json".format(name, n))), 'w') as f:
        json.dump(simulated_lcs, f,  indent=4, separators=(", ", ": "), sort_keys=True)


def load_data_Fermi():
    t_fermi = Table.read("data/lc_2008_2020.fits")

    mjd = t_fermi['tmean']
    time = (mjd - mjd.min()) * u.day
    flux = t_fermi['flux']
    err = t_fermi['flux_err']

    return time, flux, err


if __name__ == "__main__":
    output_path = "build"

    time, flux, err = load_data_Fermi()
    run_lc_simulation(time, flux, err, name="Fermi_2008_2020", n=3, output_path=output_path)







