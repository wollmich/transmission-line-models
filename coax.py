'''
@author: Ziad Hatab (zi.hatab@gmail.com)

This is an implementation of a coaxial line model based on the references [1-2].
Also, I included the option to calculate Jacobian with respect to the inputs.

The conductor impedances Za and Zb are available in two forms:
    - simple_model=False (default) : the Bessel-function solution of [2].
    - simple_model=True            : the equivalent circuit of [1].

Both models describe the conductors through their surface impedance Zs (ohms per
square). Neither model uses Zs raw: each first turns it into a per-unit-length impedance
that also carries the dc resistance and the internal inductance of the conductor. The
exact model of [2] does that with a ratio of Bessel functions, and the simple model
of [1] does it with an equivalent RL circuit that approximates the very same ratio.

Every material property takes either a single number or a 1d-array with one value 
per frequency, so a dispersion model of your own can be fed in directly. 
The geometry cannot depend on frequency.

[1] F. M. Tesche, "A Simple Model for the Line Parameters of a Lossy Coaxial
Cable Filled With a Nondispersive Dielectric,"
in IEEE Transactions on Electromagnetic Compatibility,
vol. 49, no. 1, pp. 12-17, Feb. 2007,
doi: https://doi.org/10.1109/TEMC.2006.888185

[2] S. A. Schelkunoff, "The electromagnetic theory of coaxial transmission lines
and cylindrical shields," in The Bell System Technical Journal,
vol. 13, no. 4, pp. 532-579, Oct. 1934,
doi: https://doi.org/10.1002/j.1538-7305.1934.tb00679.x
'''

# pip install numpy scipy -U
import numpy as np
from scipy.special import ive, kve  # exponentially scaled modified Bessel functions

# physical constants
mu0 = 1.25663706127e-6  # vacuum permeability in H/m (CODATA 2022)
ep0 = 8.8541878188e-12  # vacuum permittivity in F/m (CODATA 2022)
c0  = 1/np.sqrt(mu0*ep0)  # speed of light in vacuum in m/s

def Za(a, f, Zs=None, sigma=58e6, mur=1, tandm=0, simple_model=False):
    '''
    Impedance per unit length of the inner conductor. The surface impedance Zs is turned
    into a per-unit-length impedance that also carries the dc resistance and the internal
    inductance, either with the equivalent circuit of [1] or with the Bessel functions of [2].

    Parameters
    ----------
    a : number
        inner radius.
    f : number or array
        frequency in Hz.
    Zs : number or array, optional
        surface impedance of the metal in ohms per square. The default None uses the
        smooth conductor value sqrt(j*omega*mu/sigma).
    sigma : number or array, optional
        conductivity in S/m. The default 58e6 is annealed copper (100% IACS).
    mur : number or array, optional
        relative permeability. The default is 1.
    tandm : number or array, optional
        magnetic loss tangent, mu = mu0*mur*(1 - 1j*tandm). The default is 0.
    simple_model : bool, optional
        Bessel functions of [2] if False (default), equivalent circuit of [1] if True.

    Returns
    -------
    number or array
        impedance per unit length in ohms/meter.
    '''
    mu    = mur*(1 - 1j*tandm)*mu0
    omega = 2*np.pi*f

    # smooth conductor by default
    Zs = np.sqrt(1j*omega*mu/sigma) if Zs is None else Zs

    if simple_model:
        # eq. (14) in [1]: Rdc in series with (internal inductance || skin effect)
        Rdc  = 1/np.pi/a**2/sigma
        Lint = mu/8/np.pi
        Zhf  = Zs/2/np.pi/a
        return Rdc + 1j*omega*Lint*Zhf/(Zhf + 1j*omega*Lint)
    else:
        # eq. (65) in [2] (also eq. (2) in [1]).
        # ive(n,z) is I_n(z) scaled by exp(-|Re(z)|), so the scaling cancels in the ratio.
        gamma = np.sqrt(1j*omega*mu*sigma)  # propagation constant inside the metal
        return Zs/2/np.pi/a*ive(0,gamma*a)/ive(1,gamma*a)

def Zb(b, t, f, Zs=None, sigma=58e6, mur=1, tandm=0, simple_model=False):
    '''
    Impedance per unit length of the outer conductor, for the return current flowing inside
    the tube. Same construction as Za(), but with the dc resistance and the internal
    inductance of a tube of wall thickness t.

    Parameters
    ----------
    b : number
        outer radius (from inside edge).
    t : number
        thickness of outer conductor.
    f : number or array
        frequency in Hz.
    Zs : number or array, optional
        surface impedance of the metal in ohms per square. The default None uses the
        smooth conductor value sqrt(j*omega*mu/sigma).
    sigma : number or array, optional
        conductivity in S/m. The default 58e6 is annealed copper (100% IACS).
    mur : number or array, optional
        relative permeability. The default is 1.
    tandm : number or array, optional
        magnetic loss tangent, mu = mu0*mur*(1 - 1j*tandm). The default is 0.
    simple_model : bool, optional
        Bessel functions of [2] if False (default), equivalent circuit of [1] if True.

    Returns
    -------
    number or array
        impedance per unit length in ohms/meter.
    '''
    mu    = mur*(1 - 1j*tandm)*mu0
    omega = 2*np.pi*f

    c = b + t  # outer radius + thickness
    # smooth conductor by default
    Zs = np.sqrt(1j*omega*mu/sigma) if Zs is None else Zs

    if simple_model:
        # same circuit as Za(), with the dc resistance and internal inductance of a tube,
        # eqs. (12) and (13) in [1]
        Rdc  = 1/2/np.pi/b/t/sigma
        Lint = mu/2/np.pi*( c**4*np.log(c/b)/(c**2-b**2)**2 + (b**2-3*c**2)/4/(c**2-b**2))
        Zhf  = Zs/2/np.pi/b
        return Rdc + 1j*omega*Lint*Zhf/(Zhf + 1j*omega*Lint)
    else:
        # eq. (74) in [2] (also eq. (5) in [1]). I_n and K_n over- and underflow badly
        # when the wall is many skin depths thick. Using the scaled ive(n,z)=I_n(z)exp(-|Re(z)|)
        # and kve(n,z)=K_n(z)exp(z) leaves a common factor exp(|Re(y)|-x) in both the
        # numerator and the denominator, which cancels. What is left over is P, and
        # |P| <= 1 because b < c.
        gamma = np.sqrt(1j*omega*mu*sigma)  # propagation constant inside the metal
        x = gamma*b
        y = gamma*c
        P = np.exp( x + abs(x.real) - (y + abs(y.real)) )
        i1y, k1y = ive(1,y), kve(1,y)  # each needed twice below
        num = ive(0,x)*k1y*P + kve(0,x)*i1y
        den = i1y*kve(1,x)   - ive(1,x)*k1y*P
        return Zs/2/np.pi/b*num/den

def L(a, b, mur=1, tandm=0):
    '''
    External inductance per unit length, i.e. the inductance of the field
    between the two conductors. The internal inductance of the metal sits in Za() and Zb().
    It comes out complex when tandm is nonzero, the imaginary part being the magnetic loss,
    the same way a lossy dielectric would make C() complex.

    Parameters
    ----------
    a : number
        inner radius.
    b : number
        outer radius (from inside edge).
    mur : number or array, optional
        relative permeability. The default is 1.
    tandm : number or array, optional
        magnetic loss tangent, mu = mu0*mur*(1 - 1j*tandm). The default is 0.

    Returns
    -------
    number or array
        inductance per unit length in H/m.
    '''
    return mu0*mur*(1 - 1j*tandm)/2/np.pi*np.log(b/a)  # correction of [1] b/a not a/b

def C(a, b, er=1):
    '''
    Shunt capacitance per unit length of the line, set by the dielectric filling the space
    between the two conductors.

    Parameters
    ----------
    a : number
        inner radius.
    b : number
        outer radius (from inside edge).
    er : number or array, optional
        relative permittivity. The default is 1.

    Returns
    -------
    number or array
        capacitance per unit length in F/m.
    '''
    return 2*np.pi*er*ep0/np.log(b/a)  # correction of [1] b/a not a/b

def G(a, b, f, er=1, tand=0):
    '''
    Shunt conductance per unit length, i.e. the dielectric loss. It is the capacitance of
    C() scaled by omega and by the loss tangent.

    Parameters
    ----------
    a : number
        inner radius.
    b : number
        outer radius (from inside edge).
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
    return omega*C(a,b,er)*tand

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

def get_all_paras(x, coax):
    '''
    This function is used to compute the Jacobian of the parameters.
    x = [a, b, t, er, tand, sigma, mur, tandm], one row each, of length len(f). The geometry
    cannot change with frequency, so only the first entry of its rows is taken.
    '''
    coax.a     = x[0][0]
    coax.b     = x[1][0]
    coax.t     = x[2][0]
    coax.er    = x[3]
    coax.tand  = x[4]
    coax.sigma = x[5]
    coax.mur   = x[6]
    coax.tandm = x[7]
    coax.update()

    # convert complex-valued array to real-valued equivalent
    hh = lambda x: np.kron(x.real, [1,0]) + np.kron(x.imag, [0,1])

    # coax line parameters
    gamma   = coax.gamma
    h_gamma = hh(gamma)
    ereff   = coax.ereff
    h_ereff = hh(ereff)
    Z0   = coax.Z0
    h_Z0 = hh(Z0)

    # RLGC parameters
    R = coax.R
    h_R = hh(R)
    L = coax.L
    h_L = hh(L)
    G = coax.G
    h_G = hh(G)
    C = coax.C
    h_C = hh(C)

    return np.hstack((h_gamma, h_ereff, h_Z0, h_R, h_L, h_G, h_C))

class COAX:
    """
    Analytical model of coaxial cable based on [1-2] (see comments at top of this file).

    Parameters
    ----------
    a : number
        radius of inner conductor (signal) in meters.
    b : number
        radius of outer conductor (GND) from inside in meters.
    t : number
        Thickness of GND in meters.
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
    Zs_a : None, number or 1d-array
        surface impedance of the inner conductor of radius a in ohms per square. If None
        (default), it is computed from mur and sigma assuming a smooth conductor.
    Zs_b : None, number or 1d-array
        surface impedance of the outer conductor of radius b in ohms per square. If None
        (default), it is computed from mur and sigma assuming a smooth conductor.
    simple_model : bool
        If False (default), use the exact Bessel-function solution of [2]. If True, use
        the wideband circuit approximation of [1].

    Notes
    -----
    A material property given as a 1d-array holds one value per frequency; the geometry
    a, b and t is scalar. All of them are real-valued: er and mur are the real parts of the
    permittivity and the permeability, the losses go in tand and tandm, following the same
    convention, er = er*(1 - 1j*tand) and mur = mur*(1 - 1j*tandm). The only inputs allowed
    to be complex are the surface impedances Zs_a and Zs_b.
    """
    def __init__(self, a, b, t, f, er=1, tand=0, sigma=58e6, mur=1, tandm=0,
                 Zs_a=None, Zs_b=None, simple_model=False):
        self.a     = a
        self.b     = b
        self.t     = t
        self.f     = np.atleast_1d(f)
        self.er    = er
        self.tand  = tand
        self.sigma = sigma
        self.mur   = mur
        self.tandm = tandm
        self.Zs_a = Zs_a
        self.Zs_b = Zs_b
        self.simple_model = simple_model
        self.update()  # run the code

    def update(self):
        a    = self.a
        b    = self.b
        t    = self.t
        f    = self.f
        # every material property gets one value per frequency. The geometry cannot change
        # with frequency and is left alone.
        ones  = np.ones_like(f)
        er    = ones*self.er
        tand  = ones*self.tand
        sigma = ones*self.sigma
        mur   = ones*self.mur
        tandm = ones*self.tandm
        Zs_a  = None if self.Zs_a is None else ones*self.Zs_a
        Zs_b  = None if self.Zs_b is None else ones*self.Zs_b
        # everything but the surface impedances is real-valued by definition
        assert np.all(np.imag([a, b, t]) == 0), 'the geometry must be real-valued.'
        assert np.all(np.imag([er, tand, sigma, mur, tandm]) == 0), 'the material properties must be real-valued, the losses go in tand and tandm.'

        omega = 2*np.pi*f

        # per unit length parameters, eq. (6) in [1]
        self.C  = C(a,b,er)
        self.G  = G(a,b,f,er,tand)
        self.Y = self.G + 1j*omega*self.C  # shunt admittance per unit length
        
        L_val  = L(a,b,mur,tandm)
        Za_val = Za(a,f,Zs_a,sigma,mur,tandm,self.simple_model)
        Zb_val = Zb(b,t,f,Zs_b,sigma,mur,tandm,self.simple_model)
        self.Z = 1j*omega*L_val + Za_val + Zb_val # series impedance per unit length
        self.L = self.Z.imag/omega  # effective inductance per unit length
        self.R = self.Z.real  # effective resistance per unit length

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
        A user-supplied Zs_a/Zs_b is treated as fixed data, so the derivative
        with respect to mur and sigma only reflects the conductors you did not override.

        A material property given per frequency is differentiated at each frequency with
        respect to the value acting there. The shape of the jacobian stays (N,2,8) either way.
        '''
        # these are the input parameters to which the Jacobian is computed
        a     = self.a
        b     = self.b
        t     = self.t
        er    = self.er
        tand  = self.tand
        sigma = self.sigma
        mur   = self.mur
        tandm = self.tandm

        N = len(self.f)
        M = 2*N
        # one row per input parameter, one value per frequency
        ones  = np.ones_like(self.f)
        x     = [ones*p for p in [a, b, t, er, tand, sigma, mur, tandm]]
        big_J = approx_jac(x, get_all_paras, self)
        
        # undo the changes done by approx_jac(), which leaves the model at its last nudged
        # parameter set, and evaluate once more so that the outputs are the nominal ones again.
        self.a     = a
        self.b     = b
        self.t     = t
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

    # k-band connector example
    a = 1.270e-3/2      # inner radius
    b = 2.92e-3/2       # outer radius (from inside edge)
    t = 6.555e-3/2 - b  # thickness of outer conductor
    sigma = 0.01*58e6   # conductivity in S/m

    ylabels = ['real(Z0) (ohm)', 'real(ereff)']

    ## Example 1: the two conductor models against each other.
    f_wide = np.logspace(1, np.log10(40e9), 250)  # 10 Hz to 40 GHz
    coax_exact  = COAX(a, b, t, f_wide, sigma=sigma)
    coax_simple = COAX(a, b, t, f_wide, sigma=sigma, simple_model=True)

    fig, ax = plt.subplots(1, 2, figsize=(10,4), constrained_layout=True)
    for axi, ys, ye, ylabel in zip(ax,
            [coax_simple.Z0.real, coax_simple.ereff.real],
            [coax_exact.Z0.real, coax_exact.ereff.real], ylabels):
        axi.semilogx(f_wide, ys, lw=2, label='simple model [1]')
        axi.semilogx(f_wide, ye, '--', lw=2, label='exact model [2]')
        axi.set_xlabel('Frequency (Hz)')
        axi.set_ylabel(ylabel)
        axi.legend()
    fig.suptitle('Comparison of the two conductor models')

    ## Example 2: uncertainty of the exact model, over the band of the connector
    f = np.linspace(0.1, 40, 250)*1e9
    coax = COAX(a, b, t, f, sigma=sigma)

    # input uncertainties, same order as the Jacobian [a, b, t, er, tand, sigma, mur, tandm]
    da     = 0.0025e-3/2
    db     = 0.005e-3/2
    dt     = 0
    der    = 0.02
    dtand  = 0
    dsigma = 0
    dmur   = 0
    dtandm = 0
    U = np.diag([da, db, dt, der, dtand, dsigma, dmur, dtandm])**2

    coax.update_jac()  # the jacobian is not computed by default
    # J@U@J.T is the 2x2 covariance of [real, imag], so element [0,0] is the variance
    # of the real part. Use [1,1] if you want the imaginary part instead.
    std_Z0    = np.sqrt(np.array([J@U@J.T for J in coax.jac_Z0])[:,0,0])
    std_ereff = np.sqrt(np.array([J@U@J.T for J in coax.jac_ereff])[:,0,0])

    k = 2  # 95% coverage of a Gaussian distribution
    fig, ax = plt.subplots(1, 2, figsize=(10,4), constrained_layout=True)
    for axi, mu, std, ylabel in zip(ax, [coax.Z0.real, coax.ereff.real],
                                       [std_Z0, std_ereff], ylabels):
        axi.plot(f*1e-9, mu, lw=2)
        axi.fill_between(f*1e-9, mu+k*std, mu-k*std, alpha=0.3)
        axi.set_xlabel('Frequency (GHz)')
        axi.set_ylabel(ylabel)
    fig.suptitle('Exact model with 95% uncertainty coverage')

    plt.show()

# EOF
