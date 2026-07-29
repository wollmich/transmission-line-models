'''
@author: Ziad Hatab (zi.hatab@gmail.com)

This is an implementation of a coaxial line model based on the references [1-2].
Also, I included the option to calculate Jacobian with respect to the inputs.

The conductor impedances Za and Zb are available in two forms:
    - use_simple_model=True  (default) : the equivalent circuit of [1].
    - use_simple_model=False           : the Bessel-function solution of [2].

Both models describe the conductors through their surface impedance Zs (ohms per
square). Neither model uses Zs raw: each first turns it into a per-unit-length impedance 
that also carries the dc resistance and the internal inductance of the conductor. The 
exact model of [2] does that with a ratio of Bessel functions, and the simple model 
of [1] does it with an equivalent RL circuit that approximates the very same ratio. 

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
import scipy.optimize as so  # for derivative computation
from scipy.special import ive, kve  # exponentially scaled modified Bessel functions

# physical constants
mu0 = 1.25663706127e-6  # vacuum permeability in H/m (CODATA 2022)
ep0 = 8.8541878188e-12  # vacuum permittivity in F/m (CODATA 2022)
c0  = 1/np.sqrt(mu0*ep0)  # speed of light in vacuum in m/s

def Za(a, f, Zs=None, mur=1, sigma=58e6, use_simple_model=True):
    '''
    Parameters
    ----------
    a : number
        inner radius.
    f : number or array
        frequency in Hz.
    Zs : number or array, optional
        surface impedance of the metal in ohms per square. The default None uses the
        smooth conductor value sqrt(j*omega*mu/sigma).
    mur : number, optional
        relative permeability. The default is 1.
    sigma : number, optional
        conductivity in S/m. The default 58e6 is annealed copper (100% IACS).
    use_simple_model : bool, optional
        equivalent circuit of [1] if True (default), Bessel functions of [2] if False.

    Returns
    -------
    number
        impedance per unit length of the inner conductor in ohms/meter.
    '''
    mu    = mur*mu0
    omega = 2*np.pi*f

    # smooth conductor by default
    Zs = np.sqrt(1j*omega*mu/sigma) if Zs is None else Zs

    if use_simple_model:
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

def Zb(b, t, f, Zs=None, mur=1, sigma=58e6, use_simple_model=True):
    '''
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
    mur : number, optional
        relative permeability. The default is 1.
    sigma : number, optional
        conductivity in S/m. The default 58e6 is annealed copper (100% IACS).
    use_simple_model : bool, optional
        equivalent circuit of [1] if True (default), Bessel functions of [2] if False.

    Returns
    -------
    number
        impedance per unit length of the outer conductor in ohms/meter, for the return
        current flowing inside the tube.
    '''
    mu    = mur*mu0
    omega = 2*np.pi*f

    c = b + t  # outer radius + thickness
    # smooth conductor by default
    Zs = np.sqrt(1j*omega*mu/sigma) if Zs is None else Zs

    if use_simple_model:
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

def L(a, b, mur=1):
    '''
    Parameters
    ----------
    a : number
        inner radius.
    b : number
        outer radius (from inside edge).
    mur : number, optional
        relative permeability. The default is 1.

    Returns
    -------
    number
        external (high frequency) inductance per unit length in H/m.
    '''
    return mu0*mur/2/np.pi*np.log(b/a)  # correction of [1] b/a not a/b

def C(a, b, er=1):
    '''
    Parameters
    ----------
    a : number
        inner radius.
    b : number
        outer radius (from inside edge).
    er : number, optional
        relative permittivity. The default is 1.

    Returns
    -------
    number
        shunt capacitance per unit length in F/m.
    '''
    return 2*np.pi*er*ep0/np.log(b/a)  # correction of [1] b/a not a/b

def G(a, b, f, er=1, tand=0):
    '''
    Parameters
    ----------
    a : number
        inner radius.
    b : number
        outer radius (from inside edge).
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
    return omega*C(a,b,er)*tand

def get_all_paras(x, coax):
    '''
    This function is used to compute the Jacobian of the parameters
    '''
    # x = [a, b, t, mur, sigma, er, tand]
    coax.a     = x[0]
    coax.b     = x[1]
    coax.t     = x[2]
    coax.mur   = x[3]
    coax.sigma = x[4]
    coax.er    = x[5]
    coax.tand  = x[6]
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

    return np.hstack((h_gamma, h_ereff, h_Z0))

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
    mur : float number
        relative permeability
    sigma : float number
        conductivity in S/m. The default 58e6 is annealed copper (100% IACS).
    er : float number
        real-part relative permittivity.
    tand : float number
        loss tangent.
    Zs_a : None, number or 1d-array
        surface impedance of the inner conductor of radius a in ohms per square. If None
        (default), it is computed from mur and sigma assuming a smooth conductor.
    Zs_b : None, number or 1d-array
        surface impedance of the outer conductor of radius b in ohms per square. If None
        (default), it is computed from mur and sigma assuming a smooth conductor.
    use_simple_model : bool
        If True (default), use the wideband circuit approximation of [1]. If False, use
        the exact Bessel-function solution of [2].
    """
    def __init__(self, a, b, t, f, mur=1, sigma=58e6, er=1, tand=0,
                 Zs_a=None, Zs_b=None, use_simple_model=True):
        self.a     = a
        self.b     = b
        self.t     = t
        self.f     = np.atleast_1d(f)
        self.mur   = mur
        self.sigma = sigma
        self.er    = er
        self.tand  = tand
        self.Zs_a = Zs_a
        self.Zs_b = Zs_b
        self.use_simple_model = use_simple_model
        self.update()  # run the code

    def update(self):
        a    = self.a
        b    = self.b
        t    = self.t
        f    = self.f
        mur   = self.mur
        sigma = self.sigma
        er    = self.er
        tand  = self.tand

        omega = 2*np.pi*f

        # per unit length parameters, eq. (6) in [1]
        self.C  = C(a,b,er)
        self.G  = G(a,b,f,er,tand)
        self.Y = self.G + 1j*omega*self.C  # shunt admittance per unit length
        
        L_val  = L(a,b,mur)
        Za_val = Za(a,f,self.Zs_a,mur,sigma,self.use_simple_model)
        Zb_val = Zb(b,t,f,self.Zs_b,mur,sigma,self.use_simple_model)
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

    def update_jac(self):
        '''
        Updates the jacobian of the parameters with respect to the input parameters.
        A user-supplied Zs_a/Zs_b is treated as fixed data, so the derivative
        with respect to mur and sigma only reflects the conductors you did not override.
        '''
        # these are the input parameters to which the Jacobian is computed
        a     = self.a
        b     = self.b
        t     = self.t
        mur   = self.mur
        sigma = self.sigma
        er    = self.er
        tand  = self.tand

        N = len(self.f)
        M = 2*N
        x = [a, b, t, mur, sigma, er, tand]
        # Finite-difference step, proportional to each parameter so that they all get the
        # same relative nudge. Parameters that are exactly zero fall back to an absolute step.
        rel   = 1e-6
        big_J = so.approx_fprime(x, get_all_paras, rel*np.where(np.abs(x) > 0, np.abs(x), 1), self)

        # split the jacobian in correct order
        self.jac_gamma = big_J[:M].reshape((N,2,-1))
        self.jac_ereff = big_J[M:2*M].reshape((N,2,-1))
        self.jac_Z0    = big_J[2*M:3*M].reshape((N,2,-1))
        self.jac_Gamma = np.array([ np.array([[(1/2/z).real, (1j/2/z).real],[(1/2/z).imag, (1j/2/z).imag]])@Jz for z,Jz in zip(self.Z0, self.jac_Z0) ])

        # undo the changes done by the function so.approx_fprime().
        self.a     = a
        self.b     = b
        self.t     = t
        self.mur   = mur
        self.sigma = sigma
        self.er    = er
        self.tand  = tand

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
    coax_simple = COAX(a, b, t, f_wide, sigma=sigma)
    coax_exact  = COAX(a, b, t, f_wide, sigma=sigma, use_simple_model=False)

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

    ## Example 2: uncertainty of the simple model, over the band of the connector
    f = np.linspace(0.1, 40, 250)*1e9
    coax = COAX(a, b, t, f, sigma=sigma)

    # input uncertainties, same order as the Jacobian [a, b, t, mur, sigma, er, tand]
    da     = 0.0025e-3/2
    db     = 0.005e-3/2
    dt     = 0
    dmur   = 0
    dsigma = 0
    der    = 0.02
    dtand  = 0
    U = np.diag([da, db, dt, dmur, dsigma, der, dtand])**2

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
    fig.suptitle('Simple model with 95% uncertainty coverage')

    plt.show()

# EOF
