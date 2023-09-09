# Transmission Line Models

A set of scripts for analytically modeling certain transmission lines and waveguides. The scripts also provide the ability to compute Jacobian derivatives with respect to input parameters, which can be useful in evaluating linear uncertainties.

## TO-DO

- Incorporate a surface impedance description of conductors to account for surface roughness on an individual conductor basis.
- Add more models.
- Standardize the parameter style across all scripts to improve code clarity.

## Supported models

- Rectangular waveguide (see [`rw.py`][rw])
- Coaxial (see [`coax.py`][coax])
- Coplanar waveguide (see [`cpw.py`][cpw])
- Microstrip (in to-do list)
- Stripline (in to-do list)
- Grounded CPW (in to-do list)

## Code dependencies

Please check the header of each script, but in general, [`numpy`](https://github.com/numpy/numpy) and [`scipy`](https://github.com/scipy/scipy) are always used. The package [`scipy`](https://github.com/scipy/scipy) is primarily used for derivatives and special functions.

```python
python -m pip install numpy scipy -U
```

## Sample code

Below is an example of a rectangular waveguide.

```python
import numpy as np
from rw import RW # my code (in same folder as your main script)

# WR12 example
f = np.linspace(60, 90, 250)*1e9  # frequency in Hz
w = 3.0988e-3 # width in meters
h = 1.5494e-3 # height in meters
mur  = 1      # relative permeability
sr   = 0.28   # relative conductivity to copper (5.98e7 S/m)
er   = 1      # relative permittivity
tand = 0      # loss tangent

rw = RW(w=w, h=h, f=f, mur=mur, sr=sr, er=er, tand=tand)

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

## References

### _Rectangular waveguide_

- K. Lomakin, G. Gold and K. Helmreich, "Transmission line model for rectangular waveguides accurately incorporating loss effects," 2017 IEEE 21st Workshop on Signal and Power Integrity (SPI), Lake Maggiore, Italy, 2017, pp. 1-4, doi: 10.1109/SaPIW.2017.7944024. <https://ieeexplore.ieee.org/document/7944024>

- K. Lomakin, G. Gold and K. Helmreich, "Analytical Waveguide Model Precisely Predicting Loss and Delay Including Surface Roughness," in IEEE Transactions on Microwave Theory and Techniques, vol. 66, no. 6, pp. 2649-2662, June 2018, doi: 10.1109/TMTT.2018.2827383. <https://ieeexplore.ieee.org/document/8356729>

### _Coplanar waveguide_

- W. Heinrich, "Quasi-TEM description of MMIC coplanar lines including conductor-loss effects," in IEEE Transactions on Microwave Theory and Techniques, vol. 41, no. 1, pp. 45-52, Jan. 1993, doi: 10.1109/22.210228. <https://ieeexplore.ieee.org/document/210228>

- F. Schnieder, T. Tischler and W. Heinrich, "Modeling dispersion and radiation characteristics of conductor-backed CPW with finite ground width," in IEEE Transactions on Microwave Theory and Techniques, vol. 51, no. 1, pp. 137-143, Jan. 2003, doi: 10.1109/TMTT.2002.806926. <https://ieeexplore.ieee.org/document/1159676>

- G. N. Phung, U. Arz, K. Kuhlmann, R. Doerner and W. Heinrich, "Improved Modeling of Radiation Effects in Coplanar Waveguides with Finite Ground Width," 2020 50th European Microwave Conference (EuMC), 2021, pp. 404-407, doi: 10.23919/EuMC48046.2021.9338133. <https://ieeexplore.ieee.org/document/9338133>

### _Coaxial_

- F. M. Tesche, "A Simple Model for the Line Parameters of a Lossy Coaxial Cable Filled With a Nondispersive Dielectric," in IEEE Transactions on Electromagnetic Compatibility, vol. 49, no. 1, pp. 12-17, Feb. 2007, doi: 10.1109/TEMC.2006.888185. <https://ieeexplore.ieee.org/document/4106102>


[rw]: https://github.com/ZiadHatab/transmission-line-models/blob/main/rw.py
[cpw]: https://github.com/ZiadHatab/transmission-line-models/blob/main/cpw.py
[coax]: https://github.com/ZiadHatab/transmission-line-models/blob/main/coax.py