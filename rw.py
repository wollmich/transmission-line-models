'''
@author: Ziad Hatab (zi.hatab@gmail.com)

This is an implementation of a rectangular waveguide model based on the references [1-2].
Also, I included the option to calculate Jacobian with respect to the inputs.

The TE10 mode is described by the two-wire circuit of [1]: a series impedance Z1 carrying
the transversal magnetic field, and a shunt branch made of C, G and a times-unit-length
impedance Z2 carrying the longitudinal magnetic field.

Each wall pair has its own surface impedance, Zs_w for the two walls of width w and Zs_h
for the two of height h, since they are not equally rough in practice [2].

[1] K. Lomakin, G. Gold, and K. Helmreich, "Transmission line model for rectangular
waveguides accurately incorporating loss effects," 2017 IEEE 21st Workshop on Signal
and Power Integrity (SPI), Lake Maggiore, Italy, 2017, pp. 1-4,
doi: https://doi.org/10.1109/SaPIW.2017.7944024

[2] K. Lomakin, G. Gold, and K. Helmreich, "Analytical Waveguide Model Precisely
Predicting Loss and Delay Including Surface Roughness," in IEEE Transactions on
Microwave Theory and Techniques, vol. 66, no. 6, pp. 2649-2662, June 2018,
doi: https://doi.org/10.1109/TMTT.2018.2827383
'''

# pip install numpy scipy -U
import numpy as np
import scipy.optimize as so  # for derivative computation

# physical constants
mu0 = 1.25663706127e-6  # vacuum permeability in H/m (CODATA 2022)
ep0 = 8.8541878188e-12  # vacuum permittivity in F/m (CODATA 2022)
c0  = 1/np.sqrt(mu0*ep0)  # speed of light in vacuum in m/s

def Z1(h, f, Zs_w=None, mur=1, sigma=58e6):
    '''
    Parameters
    ----------
    h : number
        height of the waveguide in meters (h < w).
    f : number or array
        frequency in Hz.
    Zs_w : number or array, optional
        surface impedance of the two walls of width w (top and bottom) in ohms per square.
        The default None uses the smooth conductor value sqrt(j*omega*mu/sigma).
    mur : number, optional
        relative permeability. The default is 1.
    sigma : number, optional
        conductivity in S/m. The default 58e6 is annealed copper (100% IACS).

    Returns
    -------
    number
        series impedance per unit length in ohms/meter, eqs. (14) and (26) in [2]. Only the
        walls of width w carry longitudinal current, yet the loss scales with 1/h and not
        with w, as remarked under eq. (30) in [1], so w does not enter here.
    '''
    mu    = mur*mu0
    omega = 2*np.pi*f

    # smooth conductor by default
    Zs_w = np.sqrt(1j*omega*mu/sigma) if Zs_w is None else Zs_w

    return 1j*omega*mu + 2/h*Zs_w

def Z2(w, h, f, Zs_w=None, Zs_h=None, mur=1, sigma=58e6):
    '''
    Parameters
    ----------
    w : number
        width of the waveguide in meters.
    h : number
        height of the waveguide in meters (h < w).
    f : number or array
        frequency in Hz.
    Zs_w : number or array, optional
        surface impedance of the two walls of width w (top and bottom) in ohms per square.
        The default None uses the smooth conductor value sqrt(j*omega*mu/sigma).
    Zs_h : number or array, optional
        surface impedance of the two walls of height h (left and right) in ohms per square.
        The default None uses the smooth conductor value sqrt(j*omega*mu/sigma).
    mur : number, optional
        relative permeability. The default is 1.
    sigma : number, optional
        conductivity in S/m. The default 58e6 is annealed copper (100% IACS).

    Returns
    -------
    number
        shunt impedance times unit length in ohms*meter. This sits in the shunt path, so
        it is a times-unit-length quantity, eqs. (16) and (34) in [2].
    '''
    mu    = mur*mu0
    omega = 2*np.pi*f

    # smooth conductor by default
    Zs_w = np.sqrt(1j*omega*mu/sigma) if Zs_w is None else Zs_w
    Zs_h = np.sqrt(1j*omega*mu/sigma) if Zs_h is None else Zs_h

    return 1j*omega*mu0*w**2/np.pi**2 + 2*w**2/np.pi**2/h*Zs_w + 4*w/np.pi**2*Zs_h

def C(er=1):
    '''
    Parameters
    ----------
    er : number, optional
        relative permittivity. The default is 1.

    Returns
    -------
    number
        shunt capacitance per unit length in F/m.
    '''
    return er*ep0

def G(f, er=1, tand=0):
    '''
    Parameters
    ----------
    f : number or array
        frequency in Hz.
    er : number, optional
        relative permittivity. The default is 1.
    tand : number, optional
        loss tangent. The default is 0.

    Returns
    -------
    number
        shunt conductance per unit length in S/m.
    '''
    omega = 2*np.pi*f
    return omega*C(er)*tand

def get_all_paras(x, rw):
    '''
    This function is used to compute the Jacobian of the parameters
    '''
    # x = [w, h, mur, sigma, er, tand]
    rw.w     = x[0]
    rw.h     = x[1]
    rw.mur   = x[2]
    rw.sigma = x[3]
    rw.er    = x[4]
    rw.tand  = x[5]
    rw.update()

    # convert complex-valued array to real-valued equivalent
    hh = lambda x: np.kron(x.real, [1,0]) + np.kron(x.imag, [0,1])

    # RW line parameters
    gamma   = rw.gamma
    h_gamma = hh(gamma)
    ereff   = rw.ereff
    h_ereff = hh(ereff)
    Z0   = rw.Z0
    h_Z0 = hh(Z0)

    return np.hstack((h_gamma, h_ereff, h_Z0))

class RW:
    """
    Analytical model of rectangular waveguide (RW) based on [1-2] (see comments at top of this file).

    Parameters
    ----------
    w : number
        width in meters.
    h : number
        height in meters (h < w).
    f : number or 1d-array
        frequency in Hz.
    mur : float number
        relative permeability.
    sigma : float number
        conductivity in S/m. The default 58e6 is annealed copper (100% IACS).
    er : float number
        real-part relative permittivity.
    tand : float number
        loss tangent.
    Zs_w : None, number or 1d-array
        surface impedance of the two walls of width w in ohms per square. If None (default),
        it is computed from mur and sigma assuming a smooth conductor.
    Zs_h : None, number or 1d-array
        surface impedance of the two walls of height h in ohms per square. If None (default),
        it is computed from mur and sigma assuming a smooth conductor.
    """
    def __init__(self, w, h, f, mur=1, sigma=58e6, er=1, tand=0,
                 Zs_w=None, Zs_h=None):
        self.w     = w
        self.h     = h
        self.f     = np.atleast_1d(f)
        self.mur   = mur
        self.sigma = sigma
        self.er    = er
        self.tand  = tand
        self.Zs_w = Zs_w
        self.Zs_h = Zs_h
        self.update()  # run the code

    def update(self):
        w     = self.w
        h     = self.h
        f     = self.f
        mur   = self.mur
        sigma = self.sigma
        er    = self.er
        tand  = self.tand

        omega = 2*np.pi*f

        # per unit length parameters, eqs. (39) and (40) in [1]
        self.Z = Z1(h, f, self.Zs_w, mur, sigma)  # series impedance per unit length
        self.R = self.Z.real  # effective resistance per unit length
        self.L = self.Z.imag/omega  # effective inductance per unit length

        Z2_val = Z2(w, h, f, self.Zs_w, self.Zs_h, mur, sigma)
        self.Y = G(f,er,tand) + 1j*omega*C(er) + 1/Z2_val  # shunt admittance per unit length
        self.G = self.Y.real  # effective conductance per unit length
        self.C = self.Y.imag/omega  # effective capacitance per unit length

        # Final results
        self.gamma = np.sqrt(self.Z*self.Y)
        self.ereff = -(c0/2/np.pi/f*self.gamma)**2
        self.Z0    = np.sqrt(self.Z/self.Y)
        # set Jacobians to None if newly evaluated (you need to run update_jac() to compute them)
        self.jac_gamma = None
        self.jac_ereff = None
        self.jac_Z0    = None
        self.jac_Gamma = None

    def update_jac(self):
        '''
        Updates the jacobian of the parameters with respect to the input parameters.
        A user-supplied Zs_w/Zs_h is treated as fixed data, so the derivative with respect
        to mur and sigma only reflects the walls you did not override.
        '''
        # these are the input parameters to which the Jacobian is computed
        w     = self.w
        h     = self.h
        mur   = self.mur
        sigma = self.sigma
        er    = self.er
        tand  = self.tand

        N = len(self.f)
        M = 2*N
        x   = [w, h, mur, sigma, er, tand]
        # Finite-difference step, proportional to each parameter so that they all get the
        # same relative nudge. A single fixed step cannot suit both sigma (order 1e7) and a
        # dimension (order 1e-3). Parameters that are exactly zero fall back to an absolute step.
        rel = 1e-6
        big_J = so.approx_fprime(x, get_all_paras, rel*np.where(np.abs(x) > 0, np.abs(x), 1), self)

        # split the jacobian in correct order
        self.jac_gamma = big_J[:M].reshape((N,2,-1))
        self.jac_ereff = big_J[M:2*M].reshape((N,2,-1))
        self.jac_Z0    = big_J[2*M:3*M].reshape((N,2,-1))
        self.jac_Gamma = np.array([ np.array([[(1/2/z).real, (1j/2/z).real],[(1/2/z).imag, (1j/2/z).imag]])@Jz for z,Jz in zip(self.Z0, self.jac_Z0) ])

        # undo the changes done by the function so.approx_fprime().
        self.w     = w
        self.h     = h
        self.mur   = mur
        self.sigma = sigma
        self.er    = er
        self.tand  = tand

if __name__ == '__main__':
    import matplotlib.pyplot as plt # for plotting

    # WR12 example
    w = 3.0988e-3      # width in meters
    h = 1.5494e-3      # height in meters
    sigma = 0.28*58e6  # conductivity of brass in S/m

    ylabels = ['Attenuation (dB/m)', 'real(ereff)']
    np2db = 20*np.log10(np.e)  # nepers to dB

    f  = np.linspace(60, 90, 250)*1e9
    rw = RW(w, h, f, sigma=sigma)

    # input uncertainties, same order as the Jacobian [w, h, mur, sigma, er, tand]
    dw     = 0.5e-3/3
    dh     = 0.5e-3/3
    dmur   = 0
    dsigma = 0.05*sigma
    der    = 0
    dtand  = 0
    U = np.diag([dw, dh, dmur, dsigma, der, dtand])**2

    rw.update_jac()  # the jacobian is not computed by default
    # J@U@J.T is the 2x2 covariance of [real, imag], so element [0,0] is the variance
    # of the real part. Use [1,1] if you want the imaginary part instead.
    std_alpha = np2db*np.sqrt(np.array([J@U@J.T for J in rw.jac_gamma])[:,0,0])
    std_ereff =       np.sqrt(np.array([J@U@J.T for J in rw.jac_ereff])[:,0,0])

    k = 2  # 95% coverage of a Gaussian distribution
    fig, ax = plt.subplots(1, 2, figsize=(10,4), constrained_layout=True)
    for axi, mu, std, ylabel in zip(ax, [np2db*rw.gamma.real, rw.ereff.real],
                                       [std_alpha, std_ereff], ylabels):
        axi.plot(f*1e-9, mu, lw=2)
        axi.fill_between(f*1e-9, mu+k*std, mu-k*std, alpha=0.3)
        axi.set_xlabel('Frequency (GHz)')
        axi.set_ylabel(ylabel)
    fig.suptitle('WR12 waveguide with 95% uncertainty coverage')

    plt.show()

# EOF
