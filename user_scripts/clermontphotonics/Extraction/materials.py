# encoding utf-8
import numpy as np
from scipy.special import wofz
import json
#from enum import Enum

class Material:

    def __init__(self,permittivity):
        self.permittivity = permittivity

    def get_permittivity(self,wavelength):
        return self.permittivity

    def get_permeability(self,wavelength):
        return 1.0

class CustomFunction(Material):

    def __init__(self,permittivity_function):
        self.permittivity_function = permittivity_function

    def get_permittivity(self,wavelength):
        return self.permittivity_function(wavelength)

class ExpData(Material):
    """
    Class of materials defined by their permittivity measured for
    well defined values of the wavelength in vacuum. We make asin
    interpolation to get the most accurate value of the permittivity.
    Two lists are thus expected:
    - wavelength_list
    - permittivities (potentially complex)
    """

    def __init__(self, wavelength_list,permittivities):

        self.wavelength_list = np.array(wavelength_list, dtype = float)
        self.permittivities  = np.array(permittivities, dtype = complex)

    def get_permittivity(self, wavelength):
        return np.interp(wavelength, self.wavelength_list, self.permittivities)

class MagneticND(Material):

    """
    Magnetic, non-dispersive material, characterized by a permittivity
    and a permeabilty that do not depend on the wavelength.
    """

    def __init__(self,permittivity,permeability):
        self.permittivity = permittivity
        self.permeability  = permeability

    def get_permeability(self,wavelength):
        return self.permeability


class BrendelBormann(Material):
    """
    Material described using a Brendel Bormann model for a metal.
    """

    def __init__(self, f0,gamma0,omega_p,f,gamma,omega,sigma) -> None:
        self.f0 = f0
        self.Gamma0 = gamma0
        self.omega_p = omega_p
        self.f = np.array(f)
        self.gamma = np.array(gamma)
        self.omega = np.array(omega)
        self.sigma = np.array(sigma)

    def get_permittivity(self, wavelength):
        w = 6.62606957e-25 * 299792458 / 1.602176565e-19 / wavelength
        a = np.sqrt(w * (w + 1j * self.gamma))
        x = (a - self.omega) / (np.sqrt(2) * self.sigma)
        y = (a + self.omega) / (np.sqrt(2) * self.sigma)
        # Polarizability due to bound electrons
        chi_b = np.sum(1j * np.sqrt(np.pi) * self.f * self.omega_p ** 2 /
                       (2 * np.sqrt(2) * a * self.sigma) * (wofz(x) + wofz(y)))
        # Equivalent polarizability linked to free electrons (Drude model)
        chi_f = -self.omega_p ** 2 * self.f0 / (w * (w + 1j * self.Gamma0))
        epsilon = 1 + chi_f + chi_b
        return epsilon

def existing_materials():
    f=open("../data/material_data.json")
    database = json.load(f)
    for entree in database:
        if "info" in database[entree]:
            print(entree,"::",database[entree]["info"])
        else :
            print(entree)

# Sometimes materials can be defined not by a well known model
# like Cauchy or Sellmeier or Lorentz, but have specific formula.
# That may be convenient.

def permittivity_glass(wl):
    #epsilon=2.978645+0.008777808/(wl**2*1e-6-0.010609)+84.06224/(wl**2*1e-6-96)
    epsilon = (1.5130 - 3.169e-9*wl**2 + 3.962e3/wl**2)**2
    return epsilon

# Declare authorized functions in the database. Add the functions listed above.

authorized = {"permittivity_glass":permittivity_glass}
