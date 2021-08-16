# Light Curve Simulation

Python 3 implementation of the method for simulating light curves as proposed by [Emmanoulopoulos et al.](https://arxiv.org/abs/1305.0304) with some extras proposed by [Max-Moerbeck et al.](https://academic.oup.com/mnras/article/445/1/437/986520). The method by Emmanoulopoulos et al. is furthermore based on the method proposed by [Timmer & Koenig](http://articles.adsabs.harvard.edu/pdf/1995A%26A...300..707T).

A detailed documentation about the methods is given in [this PDF](documentation_LC_simulation.pdf).

Run the example script with

```python
python scripts/simulate_lightcurves.py
```

To create a light curve object, you need at least a sequence of times and flux values.
Additionally, a bin with `tbin` for the interpolation and error values can be given.
`tbin` must be given in units of days or seconds!
`time` and `flux` 

```python
from astropy.table import Table
import astropy.units as u
from emmanoulopoulos.lightcurve import LC

t_fermi = Table.read("data/lc_2008_2020.fits")

mjd = t_fermi['tmean']
time = (mjd - mjd.min()) * u.day
flux = t_fermi['flux']

lc_original = LC(time, flux, tbin=10 * u.day)
```

Fit the PSD and PDF from the original light curve:
```python
lc_original.fit_PSD()
lc_original.fit_PDF()
```

Create a light curve sampler and sample a new lightcurve from the original light curve:
```python
from emmanoulopoulos.emmanoulopoulos_lc_simulation import Emmanoulopoulos_Sampler

Emma = Emmanoulopoulos_Sampler()
lc_sim = Emma.sample_from_lc(lc_original)
```


