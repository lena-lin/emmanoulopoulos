from emmanoulopoulos.emmanoulopoulos_lc_simulation import Emmanoulopoulos_Sampler
from emmanoulopoulos.lightcurve import LC
from astropy.table import Table
from pathlib import Path
import astropy.units as u
import json
from json import encoder
from tqdm import tqdm

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

    print(pdf_parameter)


    simulated_lcs = {}
    Emma = Emmanoulopoulos_Sampler()
    for i in tqdm(range(n)):
        lc_sim = Emma.sample_from_lc(lc)
        simulated_lcs[i] = list(lc_sim.original_flux)

    with open(str(Path(output_path, "sim_lc_{}_n{}.json".format(name, n))), 'w') as f:
        json.dump(simulated_lcs, f,  indent=4, separators=(", ", ": "), sort_keys=True)


def load_data_VLBA_43(comp):
    t_vlba = Table.read('/home/llinhoff/Documents/Thesis/Data/Radio_Components_Flux/flux_components_43_calibrated.fits')

    mjd = t_vlba['MJD']
    time = (mjd - mjd.min()) * u.day
    flux = t_vlba[comp]
    err = t_vlba[f'{comp}_ERR']

    return time, flux, err


def load_data_Fermi():
    t_fermi = Table.read("/home/llinhoff/Documents/FERMI/Analysis/LC_2008_2020/lc_2008_2020.fits")

    mjd = t_fermi['tmean']
    time = (mjd - mjd.min()) * u.day
    flux = t_fermi['flux']
    err = t_fermi['flux_err']

    return time, flux, err


@click.command()
@click.option(
    '--waveband',
    type=click.choice(["C1", "C3", "Fermi"]),
    required=True
)
@click.option(
    '-output_path',
    type=click.Path(file_okay=False, dir_okay=True)
)
def main(waveband):
    if waveband == "C1":
        time, flux, err = load_data_VLBA_43(comp)

run_lc_simulation(time, flux, err, name=f"VLBA_43_{comp}_calibrated", n=10, output_path=output_path)



if __name__ == "__main__":
    output_path = "/home/llinhoff/Documents/Thesis/Data/LC_Correlation/LC_Sim"
    n = 3
    name = "VLBA_43_C1_calibrated"
    for comp in ["C1", "C3"]:
        time, flux, err = load_data_VLBA_43(comp)
        run_lc_simulation(time, flux, err, name=f"VLBA_43_{comp}_calibrated", n=10, output_path=output_path)

    time, flux, err = load_data_Fermi()
    # run_lc_simulation(time, flux, err, name="Fermi_2008_2020", n=10, output_path=output_path)







