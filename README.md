# Transmission Line Models

A set of scripts for analytically modeling certain transmission lines and waveguides. The scripts also provide the ability to compute Jacobian derivatives with respect to input parameters, which can be useful in evaluating linear uncertainties.

## TO-DO

- Incorporate a surface impedance description of conductors on an individual conductor basis (done for [`rw.py`][rw] and [`coax.py`][coax]).
- Add more models.
- Standardize the parameter style across all scripts to improve code clarity.

## Supported models

- Rectangular waveguide (see [`rw.py`][rw])
- Coaxial (see [`coax.py`][coax])
- Coplanar waveguide (see [`cpw.py`][cpw])
- Microstrip (see [`ms.py`][ms])
- Stripline (in to-do list)
- Grounded CPW (in to-do list)

## Code dependencies

Please check the header of each script, but in general, [`numpy`](https://github.com/numpy/numpy) is always used, and [`scipy`](https://github.com/scipy/scipy) wherever special functions are needed. The example at the bottom of each script also uses [`matplotlib`](https://matplotlib.org/) for plotting.

```python
python -m pip install numpy scipy matplotlib -U
```

## Sample code

Below is an example of a rectangular waveguide.

```python
import numpy as np
from rw import RW # my code (in same folder as your main script)

# WR12 example
f = np.linspace(60, 90, 250)*1e9  # frequency in Hz
w     = 3.0988e-3  # width in meters
h     = 1.5494e-3  # height in meters
mur   = 1          # relative permeability
sigma = 0.28*58e6  # conductivity in S/m (58e6 is annealed copper, 100% IACS)
er    = 1          # relative permittivity
tand  = 0          # loss tangent

rw = RW(w=w, h=h, f=f, mur=mur, sigma=sigma, er=er, tand=tand)

gamma = rw.gamma  # propagation constant
ereff = rw.ereff  # complex-valued effective permittivity
Z0    = rw.Z0     # characteristic impedance

rw.update_jac()  # compute the jacobian (by default not computed)
# The jacobian is calculated with respect to all input parameters 
# in same order as defined in the class RW.
jac_gamma = rw.jac_gamma
jac_ereff = rw.jac_ereff
jac_Z0    = rw.jac_Z0
```

## Frequency-dependent material properties

Each material property may be a single number or a 1D array with one value per frequency. The geometry must be independent of frequency. The Jacobian is computed at each frequency with respect to the corresponding parameter value.

```python
f    = np.linspace(60, 90, 250)*1e9
er   = 3.0 - 0.02*np.log(f/1e9)  # one value per frequency
tand = 0.002 + 1e-14*f           # likewise
rw   = RW(w=w, h=h, f=f, sigma=sigma, er=er, tand=tand)
```

## References

### _Rectangular waveguide_

[1] K. Lomakin, G. Gold, and K. Helmreich, "Transmission line model for rectangular waveguides accurately incorporating loss effects," 2017 IEEE 21st Workshop on Signal and Power Integrity (SPI), Lake Maggiore, Italy, 2017, pp. 1-4, doi: [10.1109/SaPIW.2017.7944024](https://doi.org/10.1109/SaPIW.2017.7944024).

[2] K. Lomakin, G. Gold, and K. Helmreich, "Analytical Waveguide Model Precisely Predicting Loss and Delay Including Surface Roughness," IEEE Transactions on Microwave Theory and Techniques, vol. 66, no. 6, pp. 2649-2662, June 2018, doi: [10.1109/TMTT.2018.2827383](https://doi.org/10.1109/TMTT.2018.2827383).

### _Coplanar waveguide_

[1] W. Heinrich, "Quasi-TEM description of MMIC coplanar lines including conductor-loss effects," IEEE Transactions on Microwave Theory and Techniques, vol. 41, no. 1, pp. 45-52, Jan. 1993, doi: [10.1109/22.210228](https://doi.org/10.1109/22.210228).

[2] F. Schnieder, T. Tischler, and W. Heinrich, "Modeling dispersion and radiation characteristics of conductor-backed CPW with finite ground width," IEEE Transactions on Microwave Theory and Techniques, vol. 51, no. 1, pp. 137-143, Jan. 2003, doi: [10.1109/TMTT.2002.806926](https://doi.org/10.1109/TMTT.2002.806926).

[3] G. N. Phung, U. Arz, K. Kuhlmann, R. Doerner, and W. Heinrich, "Improved Modeling of Radiation Effects in Coplanar Waveguides with Finite Ground Width," 2020 50th European Microwave Conference (EuMC), 2021, pp. 404-407, doi: [10.23919/EuMC48046.2021.9338133](https://doi.org/10.23919/EuMC48046.2021.9338133).

[4] B. Eckmann, M. Wollensack, L. Ausden, J. Hoffmann, and M. Zeier, "VNA Tools: Material Parameters Extraction from On-Wafer CPW S-Parameter Measurements," 2026 106th ARFTG Microwave Measurement Conference (ARFTG), 2026, doi: [10.1109/ARFTG68976.2026.11475009](https://doi.org/10.1109/ARFTG68976.2026.11475009).

### _Coaxial_

[1] F. M. Tesche, "A Simple Model for the Line Parameters of a Lossy Coaxial Cable Filled With a Nondispersive Dielectric," IEEE Transactions on Electromagnetic Compatibility, vol. 49, no. 1, pp. 12-17, Feb. 2007, doi: [10.1109/TEMC.2006.888185](https://doi.org/10.1109/TEMC.2006.888185).

[2] S. A. Schelkunoff, "The electromagnetic theory of coaxial transmission lines and cylindrical shields," The Bell System Technical Journal, vol. 13, no. 4, pp. 532-579, Oct. 1934, doi: [10.1002/j.1538-7305.1934.tb00679.x](https://doi.org/10.1002/j.1538-7305.1934.tb00679.x).

### _Microstrip_

[1] F. Schnieder and W. Heinrich, "Model of thin-film microstrip line for circuit design," IEEE Transactions on Microwave Theory and Techniques, vol. 49, no. 1, pp. 104-110, Jan. 2001, doi: [10.1109/22.899967](https://doi.org/10.1109/22.899967).

## License

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)

[rw]: https://github.com/ZiadHatab/transmission-line-models/blob/main/rw.py
[cpw]: https://github.com/ZiadHatab/transmission-line-models/blob/main/cpw.py
[coax]: https://github.com/ZiadHatab/transmission-line-models/blob/main/coax.py
[ms]: https://github.com/ZiadHatab/transmission-line-models/blob/main/ms.py