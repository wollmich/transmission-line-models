'''
@author: Ziad Hatab (zi.hatab@gmail.com)

This is an implementation of a rectangular waveguide model based on the references [1-2].
Also, I included the option to calculate Jacobian with respect to the inputs.

The TE10 mode is described by the two-wire circuit of [1]: a series impedance Z1 carrying
the transversal magnetic field, and a shunt branch made of C, G and a times-unit-length
impedance Z2 carrying the longitudinal magnetic field.

Each wall pair has its own surface impedance, Zs_w for the two walls of width w and Zs_h
for the two of height h, since they are not equally rough in practice [2].

Every material property takes either a single number or a 1d-array with one value 
per frequency, so a dispersion model of your own can be fed in directly. 
The geometry cannot depend on frequency.

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

# physical constants
mu0 = 1.25663706127e-6  # vacuum permeability in H/m (CODATA 2022)
ep0 = 8.8541878188e-12  # vacuum permittivity in F/m (CODATA 2022)
c0  = 1/np.sqrt(mu0*ep0)  # speed of light in vacuum in m/s

def Z1(h, f, Zs_w=None, sigma=58e6, mur=1, tandm=0):
    '''
    Series impedance per unit length of the TE10 two-wire circuit, the branch carrying the
    transversal magnetic field, eqs. (14) and (26) in [2]. Only the walls of width w carry
    longitudinal current, yet the loss scales with 1/h and not with w, as remarked under
    eq. (30) in [1], so w does not enter here.

    Parameters
    ----------
    h : number
        height of the waveguide in meters (h < w).
    f : number or array
        frequency in Hz.
    Zs_w : number or array, optional
        surface impedance of the two walls of width w (top and bottom) in ohms per square.
        The default None uses the smooth conductor value sqrt(j*omega*mu/sigma).
    sigma : number or array, optional
        conductivity in S/m. The default 58e6 is annealed copper (100% IACS).
    mur : number or array, optional
        relative permeability. The default is 1.
    tandm : number or array, optional
        magnetic loss tangent, mu = mu0*mur*(1 - 1j*tandm). The default is 0.

    Returns
    -------
    number or array
        series impedance per unit length in ohms/meter.
    '''
    mu    = mur*(1 - 1j*tandm)*mu0
    omega = 2*np.pi*f

    # smooth conductor by default
    Zs_w = np.sqrt(1j*omega*mu/sigma) if Zs_w is None else Zs_w

    return 1j*omega*mu + 2/h*Zs_w

def Z2(w, h, f, Zs_w=None, Zs_h=None, sigma=58e6, mur=1, tandm=0):
    '''
    Shunt impedance of the TE10 two-wire circuit, the branch carrying the longitudinal
    magnetic field, eqs. (16) and (34) in [2]. It sits in the shunt path, so it is a
    times-unit-length quantity and not a per-unit-length one.

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
    sigma : number or array, optional
        conductivity in S/m. The default 58e6 is annealed copper (100% IACS).
    mur : number or array, optional
        relative permeability. The default is 1.
    tandm : number or array, optional
        magnetic loss tangent, mu = mu0*mur*(1 - 1j*tandm). The default is 0.

    Returns
    -------
    number or array
        shunt impedance times unit length in ohms*meter.
    '''
    mu    = mur*(1 - 1j*tandm)*mu0
    omega = 2*np.pi*f

    # smooth conductor by default
    Zs_w = np.sqrt(1j*omega*mu/sigma) if Zs_w is None else Zs_w
    Zs_h = np.sqrt(1j*omega*mu/sigma) if Zs_h is None else Zs_h

    return 1j*omega*mu0*w**2/np.pi**2 + 2*w**2/np.pi**2/h*Zs_w + 4*w/np.pi**2*Zs_h

def C(er=1):
    '''
    Shunt capacitance per unit length of the waveguide, set by the dielectric filling it.

    Parameters
    ----------
    er : number or array, optional
        relative permittivity. The default is 1.

    Returns
    -------
    number or array
        capacitance per unit length in F/m.
    '''
    return er*ep0

def G(f, er=1, tand=0):
    '''
    Shunt conductance per unit length, i.e. the dielectric loss. It is the capacitance of
    C() scaled by omega and by the loss tangent.

    Parameters
    ----------
    f : number or array
        frequency in Hz.
    er : number or array, optional
        relative permittivity. The default is 1.
    tand : number or array, optional
        loss tangent. The default is 0.

    Returns
    -------
    number or array
        conductance per unit length in S/m.
    '''
    omega = 2*np.pi*f
    return omega*C(er)*tand

def approx_jac(x, fun, obj, rel=1e-6):
    '''
    Forward-difference jacobian of fun(x,obj), where x holds one row per input parameter
    and one value per frequency. Replaces scipy.optimize.approx_fprime, which only takes
    scalar parameters.
    '''
    f0 = fun(x, obj)
    J  = np.empty((len(f0), len(x)))
    for k, row in enumerate(x):
        h  = rel*np.where(np.abs(row) > 0, np.abs(row), 1)
        xp = list(x)
        xp[k] = row + h
        h  = (xp[k] - row).real  # the step actually taken, after rounding
        # the step repeats over real and imaginary part, and over every output block
        J[:,k] = (fun(xp, obj) - f0)/np.tile(np.repeat(h, 2), len(f0)//len(row)//2)
    return J

def get_all_paras(x, rw):
    '''
    This function is used to compute the Jacobian of the parameters.
    x = [w, h, er, tand, sigma, mur, tandm], one row each, of length len(f). The geometry
    cannot change with frequency, so only the first entry of its rows is taken.
    '''
    rw.w     = x[0][0]
    rw.h     = x[1][0]
    rw.er    = x[2]
    rw.tand  = x[3]
    rw.sigma = x[4]
    rw.mur   = x[5]
    rw.tandm = x[6]
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

    # RLGC parameters
    R = rw.R
    h_R = hh(R)
    L = rw.L
    h_L = hh(L)
    G = rw.G
    h_G = hh(G)
    C = rw.C
    h_C = hh(C)

    return np.hstack((h_gamma, h_ereff, h_Z0, h_R, h_L, h_G, h_C))

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
    er : number or 1d-array
        real-part relative permittivity.
    tand : number or 1d-array
        loss tangent.
    sigma : number or 1d-array
        conductivity in S/m. The default 58e6 is annealed copper (100% IACS).
    mur : number or 1d-array
        relative permeability.
    tandm : number or 1d-array
        magnetic loss tangent.
    Zs_w : None, number or 1d-array
        surface impedance of the two walls of width w in ohms per square. If None (default),
        it is computed from mur and sigma assuming a smooth conductor.
    Zs_h : None, number or 1d-array
        surface impedance of the two walls of height h in ohms per square. If None (default),
        it is computed from mur and sigma assuming a smooth conductor.

    Notes
    -----
    A material property given as a 1d-array holds one value per frequency; the geometry
    w and h is scalar. All of them are real-valued: er and mur are the real parts of the
    permittivity and the permeability, the losses go in tand and tandm, following the same
    convention, er = er*(1 - 1j*tand) and mur = mur*(1 - 1j*tandm). The only inputs allowed
    to be complex are the surface impedances Zs_w and Zs_h.
    """
    def __init__(self, w, h, f, er=1, tand=0, sigma=58e6, mur=1, tandm=0,
                 Zs_w=None, Zs_h=None):
        self.w     = w
        self.h     = h
        self.f     = np.atleast_1d(f)
        self.er    = er
        self.tand  = tand
        self.sigma = sigma
        self.mur   = mur
        self.tandm = tandm
        self.Zs_w = Zs_w
        self.Zs_h = Zs_h
        self.update()  # run the code

    def update(self):
        w     = self.w
        h     = self.h
        f     = self.f
        # every material property gets one value per frequency. The geometry cannot change
        # with frequency and is left alone.
        ones  = np.ones_like(f)
        er    = ones*self.er
        tand  = ones*self.tand
        sigma = ones*self.sigma
        mur   = ones*self.mur
        tandm = ones*self.tandm
        Zs_w  = None if self.Zs_w is None else ones*self.Zs_w
        Zs_h  = None if self.Zs_h is None else ones*self.Zs_h
        # everything but the surface impedances is real-valued by definition
        assert np.all(np.imag([w, h]) == 0), 'the geometry must be real-valued.'
        assert np.all(np.imag([er, tand, sigma, mur, tandm]) == 0), 'the material properties must be real-valued, the losses go in tand and tandm.'

        omega = 2*np.pi*f

        # per unit length parameters, eqs. (39) and (40) in [1]
        self.Z = Z1(h, f, Zs_w, sigma, mur, tandm)  # series impedance per unit length
        self.R = self.Z.real  # effective resistance per unit length
        self.L = self.Z.imag/omega  # effective inductance per unit length

        Z2_val = Z2(w, h, f, Zs_w, Zs_h, sigma, mur, tandm)
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
        self.jac_R     = None
        self.jac_L     = None
        self.jac_G     = None
        self.jac_C     = None

    def update_jac(self):
        '''
        Updates the jacobian of the parameters with respect to the input parameters.
        A user-supplied Zs_w/Zs_h is treated as fixed data, so the derivative with respect
        to mur and sigma only reflects the walls you did not override.

        A material property given per frequency is differentiated at each frequency with
        respect to the value acting there. The shape of the jacobian stays (N,2,7) either way.
        '''
        # these are the input parameters to which the Jacobian is computed
        w     = self.w
        h     = self.h
        er    = self.er
        tand  = self.tand
        sigma = self.sigma
        mur   = self.mur
        tandm = self.tandm

        N = len(self.f)
        M = 2*N
        # one row per input parameter, one value per frequency
        ones  = np.ones_like(self.f)
        x     = [ones*p for p in [w, h, er, tand, sigma, mur, tandm]]
        big_J = approx_jac(x, get_all_paras, self)

        # undo the changes done by approx_jac(), which leaves the model at its last nudged
        # parameter set, and evaluate once more so that the outputs are the nominal ones again.
        self.w     = w
        self.h     = h
        self.er    = er
        self.tand  = tand
        self.sigma = sigma
        self.mur   = mur
        self.tandm = tandm
        self.update()

        # split the jacobian in correct order
        self.jac_gamma = big_J[:M].reshape((N,2,-1))
        self.jac_ereff = big_J[M:2*M].reshape((N,2,-1))
        self.jac_Z0    = big_J[2*M:3*M].reshape((N,2,-1))
        self.jac_Gamma = np.array([ np.array([[(1/2/z).real, (1j/2/z).real],[(1/2/z).imag, (1j/2/z).imag]])@Jz for z,Jz in zip(self.Z0, self.jac_Z0) ])

        self.jac_R     = big_J[3*M:4*M:2].reshape((N,1,-1))
        self.jac_L     = big_J[4*M:5*M:2].reshape((N,1,-1))
        self.jac_G     = big_J[5*M:6*M:2].reshape((N,1,-1))
        self.jac_C     = big_J[6*M:7*M:2].reshape((N,1,-1))

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

    # input uncertainties, same order as the Jacobian [w, h, er, tand, sigma, mur, tandm]
    dw     = 0.5e-3/3
    dh     = 0.5e-3/3
    der    = 0
    dtand  = 0
    dsigma = 0.05*sigma
    dmur   = 0
    dtandm = 0
    U = np.diag([dw, dh, der, dtand, dsigma, dmur, dtandm])**2

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
