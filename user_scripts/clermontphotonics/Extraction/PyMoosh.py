import numpy as np
import scipy.optimize as optim
from materials import *
from math import *
import sys
import copy

class Structure:
    """Each instance of Structure describes a multilayer completely.
    This includes the materials the multilayer is made of and the
    thickness of each layer.

    Args:
        materials (list) : a list of material definition
        layer_type (list) : how the different materials are stacked
        thickness (list) : thickness of each layer in nm

    Materials can be defined in the list :materials:
    -by giving their permittivity as a real number for non dispersive materials
    -by giving a list of two floats. The first is the permittivity and the
    second one the permeability. It is considered non dispersive.
    -by giving its name for dispersive materials that are in the database.
    -by giving a custom permittivity function, taking as an argument the
    wavelength in nanometer.

    .. warning: the working wavelength always is given in nanometers.

    Example: [1.,'Si','Au'] means we will use a material with a permittivity
    of 1. (air), silicon and gold.

    Each material can then be refered to by its position in the list :materials:
    The list layer_type is used to describe how the different materials are
    placed in the structure, using the place of each material in the list
    :materials:

    Example: [0,1,2,0] describes a superstrate made of air (permittivity 1.),
    on top of Si, on top of gold and a substrate made of air.

    The thickness of each layer is given in the :thickness: list, in nanometers
    typically. The thickness of the superstrate is assumed to be zero by most
    of the routines (like :coefficient:, :absorption:) so that the first
    interface is considered as the phase reference. The reflection coefficient
    of the structure will thus never be impacted by the thickness of the first
    layer.For other routines (like :field:), this thickness define the part
    of the superstrate (or substrate) that must be represented on the figure.

    Example: [0,200,300,500] actually refers to an infinite air superstrate but
    non of it will be represented, a 200 nm thick silicon layer, a 300 nm thick
    gold layer and an infinite substrate, of which a thickness of 500 nm will be
    represented is asked.
    """

    def __init__(self, materials, layer_type, thickness,verbose = True):

        materials_final=list()
        if verbose :
            print("List of materials:")
        for mat in materials:
            if issubclass(mat.__class__,Material):
                materials_final.append(mat)
                if verbose :
                    print("Object:",mat.__class__.__name__)
            elif isinstance(mat,float) or isinstance(mat,complex):
                new_mat = Material(mat)
                materials_final.append(new_mat)
                if verbose :
                    print("Simple, non dispersive: epsilon=",mat)
            elif isinstance(mat,list):
                newmat = MagneticND(mat[0],mat[1])
                materials_final.append(new_mat)
                if verbose :
                    print("Magnetic, non dispersive: epsilon=", mat[0]," mu=",mat[1])
            elif mat.__class__.__name__ == 'function':
                newmat = CustomFunction(mat)
                materials_final.append(new_mat)
                if verbose :
                    print("Custom dispersive material. Epsilon=",mat.__name__,"(wavelength in nm)")
            elif isinstance(mat,str):
                f=open("../data/material_data.json")
                database = json.load(f)
                if mat in database:
                    if verbose :
                        print("Database material:",mat)
                    material_data = database[mat]
                    model = material_data["model"]
#                    match model:
#                    case "ExpData":
                    if model == "ExpData":
                        wl=np.array(material_data["wavelength_list"])
                        epsilon = np.array(material_data["permittivities"])
                        if "permittivities_imag" in material_data:
                            epsilon = epsilon + 1j*np.array(material_data["permittivities_imag"])
                        new_mat= ExpData(wl,epsilon)
                        materials_final.append(new_mat)
#                    case "BrendelBormann"
                    elif model == "BrendelBormann":
                        f0 = material_data["f0"]
                        Gamma0 = material_data["Gamma0"]
                        omega_p = material_data["omega_p"]
                        ff = material_data["f"]
                        gamma = material_data["Gamma"]
                        omega = material_data["omega"]
                        sigma = material_data["sigma"]
                        new_mat = BrendelBormann(f0,Gamma0,omega_p,ff,gamma,omega,sigma)
                        materials_final.append(new_mat)
#                    case "CustomFunction":
                    elif model == "CustomFunction":
                        permittivity = material_data["function"]
                        new_mat = CustomFunction(authorized[permittivity])
                        materials_final.append(new_mat)
#                    case _:
                    else:
                        print(model," not an existing model.")
                        #sys.exit()
                else:
                    print(mat,"Unknown material")
                    print("Known materials:\n",existing_materials())
                    #sys.exit()
            else:
                print("Whhaaaaat ? That has nothing to do here :",mat)
#                sys.exit( )
        self.materials = materials_final
        self.layer_type = layer_type
        self.thickness = thickness

    def polarizability(self, wavelength):
        """ Computes the actual permittivity of each material considered in
        the structure. This method is called before each calculation.

        Args:
            wavelength (float): the working wavelength (in nanometers)
        """

        # Create empty mu and epsilon arrays
        mu = np.ones_like(self.materials, dtype=complex)
        epsilon = np.ones_like(self.materials, dtype=complex)
        # Loop over all materials
        for k in range(len(self.materials)):
            # Populate epsilon and mu arrays from the material.
            material = self.materials[k]
            epsilon[k] = material.get_permittivity(wavelength)
            mu[k] = material.get_permeability(wavelength)

        return epsilon, mu


class Beam:
    """ An object of the class contains all the parameters defining an incident
    beam. At initialization, a few messages will be displayed to inform the
    user.

    Args:
        wavelength (float): Wavelength in vacuum in nanometers
        incidence (float): Incidence angle in radians
        polarization (int) : '0' for TE polarization, TM otherwise
        waist (float): waist of the incident beam along the $x$ direction

    """

    def __init__(self, wavelength, incidence, polarization, horizontal_waist):
        self.wavelength = wavelength
        self.incidence = incidence
        tmp = incidence * 180 / np.pi
        print("Incidence in degrees:", tmp)
        self.polarization = polarization
        if (polarization == 0):
            print("E//, TE, s polarization")
        else:
            print("H//, TM, p polarization")
        self.waist = horizontal_waist


class Window:
    """An object containing all the parameters defining the spatial domain
    which is represented.

    Args:
        width (float): width of the spatial domain (in nm)
        beam_relative_position (float): relative position of the beam center
        horizontal_pixel_size (float): size in nm of a pixel, horizontally
        vertical_pixel_size (float): size in nm of a pixel, vertically

    The number of pixel for each layer will be computed later, but the number of
    pixel horizontally is computed and stored in nx.

    The position of the center of the beam is automatically deduced from
    the relative position: 0 means complete left of the domain, 1 complete
    right and 0.5 in the middle of the domaine, of course.
    """

    def __init__(self, width, beam_relative_position, horizontal_pixel_size,
                 vertical_pixel_size):
        self.width = width
        self.C = beam_relative_position
        self.ny = 0
        self.px = float(horizontal_pixel_size)
        self.py = float(vertical_pixel_size)
        self.nx = int(np.floor(width / self.px))
        print("Pixels horizontally:", self.nx)


""" There we go for the most fundamental functions... """


def cascade(A, B):
    """
    This function takes two 2x2 matrixes A and B, that are assumed to be scattering matrixes
    and combines them assuming A is the "upper" one, and B the "lower" one, physically.
    The result is a 2x2 scattering matrix.

    Args:
        A (2x2 numpy array):
        B (2x2 numpy array):

    """
    t = 1 / (1 - B[0, 0] * A[1, 1])
    S = np.zeros((2, 2), dtype=complex)
    S[0, 0] = A[0, 0] + A[0, 1] * B[0, 0] * A[1, 0] * t
    S[0, 1] = A[0, 1] * B[0, 1] * t
    S[1, 0] = B[1, 0] * A[1, 0] * t
    S[1, 1] = B[1, 1] + A[1, 1] * B[0, 1] * B[1, 0] * t
    return (S)


def coefficient(struct, wavelength, incidence, polarization):
    """
    This function computes the reflection and transmission coefficients
    of the structure.

    Args:
        struct (Structure): belongs to the Structure class
        wavelength (float): wavelength of the incidence light (in nm)
        incidence (float): incidence angle in radians
        polarization (float): 0 for TE, 1 (or anything) for TM

    returns:
        r (complex): reflection coefficient, phase origin at first interface
        t (complex): transmission coefficient
        R (float): Reflectance (energy reflection)
        T (float): Transmittance (energie transmission)


    R and T are the energy coefficients (real quantities)

    .. warning: The transmission coefficients have a meaning only if the lower medium
    is lossless, or they have no true meaning.
    """
    # In order to get a phase that corresponds to the expected reflected coefficient,
    # we make the height of the upper (lossless) medium vanish. It changes only the
    # phase of the reflection coefficient.

    # The medium may be dispersive. The permittivity and permability of each
    # layer has to be computed each time.
    Epsilon, Mu = struct.polarizability(wavelength)
    thickness = copy.deepcopy(struct.thickness)
    # In order to ensure that the phase reference is at the beginning
    # of the first layer.
    thickness[0] = 0
    Type = struct.layer_type
    # The boundary conditions will change when the polarization changes.
    if polarization == 0:
        f = Mu
    else:
        f = Epsilon
    # Wavevector in vacuum.
    k0 = 2 * np.pi / wavelength
    # Number of layers
    g = len(struct.layer_type)
    # Wavevector k_x, horizontal
    alpha = np.sqrt(Epsilon[Type[0]] * Mu[Type[0]]) * k0 * np.sin(incidence)
    # Computation of the vertical wavevectors k_z
    gamma = np.sqrt(
        Epsilon[Type] * Mu[Type] * k0 ** 2 - np.ones(g) * alpha ** 2)
    # Be cautious if the upper medium is a negative index one.
    if np.real(Epsilon[Type[0]]) < 0 and np.real(Mu[Type[0]]) < 0:
        gamma[0] = -gamma[0]

    # Changing the determination of the square root to achieve perfect stability
    if g > 2:
        gamma[1:g - 2] = gamma[1:g - 2] * (
                    1 - 2 * (np.imag(gamma[1:g - 2]) < 0))
    # Outgoing wave condition for the last medium
    if np.real(Epsilon[Type[g - 1]]) < 0 and np.real(
            Mu[Type[g - 1]]) < 0 and np.real(np.sqrt(Epsilon[Type[g - 1]] * Mu[
        Type[g - 1]] * k0 ** 2 - alpha ** 2)) != 0:
        gamma[g - 1] = -np.sqrt(
            Epsilon[Type[g - 1]] * Mu[Type[g - 1]] * k0 ** 2 - alpha ** 2)
    else:
        gamma[g - 1] = np.sqrt(
            Epsilon[Type[g - 1]] * Mu[Type[g - 1]] * k0 ** 2 - alpha ** 2)
    T = np.zeros(((2 * g, 2, 2)), dtype=complex)

    # first S matrix
    T[0] = [[0, 1], [1, 0]]
    for k in range(g - 1):
        # Layer scattering matrix
        t = np.exp((1j) * gamma[k] * thickness[k])
        T[2 * k + 1] = [[0, t], [t, 0]]
        # Interface scattering matrix
        b1 = gamma[k] / f[Type[k]]
        b2 = gamma[k + 1] / f[Type[k + 1]]
        T[2 * k + 2] = [[(b1 - b2) / (b1 + b2), 2 * b2 / (b1 + b2)],
                        [2 * b1 / (b1 + b2), (b2 - b1) / (b1 + b2)]]
    t = np.exp((1j) * gamma[g - 1] * thickness[g - 1])
    T[2 * g - 1] = [[0, t], [t, 0]]
    # Once the scattering matrixes have been prepared, now let us combine them
    A = np.zeros(((2 * g - 1, 2, 2)), dtype=complex)
    A[0] = T[0]

    for j in range(len(T) - 2):
        A[j + 1] = cascade(A[j], T[j + 1])
    # reflection coefficient of the whole structure
    r = A[len(A) - 1][0, 0]
    # transmission coefficient of the whole structure
    t = A[len(A) - 1][1, 0]
    # Energy reflexion coefficient;
    R = np.real(abs(r) ** 2)
    # Energy transmission coefficient;
    T = np.real(
        abs(t) ** 2 * gamma[g - 1] * f[Type[0]] / (gamma[0] * f[Type[g - 1]]))

    return r, t, R, T


# def fcoefficient(struct,wavelength)
#    '''Computation of the reflection coefficient of the structure using
#    the formalism of impedances...

def absorption(struct, wavelength, incidence, polarization):
    """
    This function computes the percentage of the incoming energy
    that is absorbed in each layer when the structure is illuminated
    by a plane wave.

    Args:
        struct (Structure): belongs to the Structure class
        wavelength (float): wavelength of the incidence light (in nm)
        incidence (float): incidence angle in radians
        polarization (float): 0 for TE, 1 (or anything) for TM

    returns:
        absorb (numpy array): absorption in each layer
        r (complex): reflection coefficient, phase origin at first interface
        t (complex): transmission coefficient
        R (float): Reflectance (energy reflection)
        T (float): Transmittance (energie transmission)
    R and T are the energy coefficients (real quantities)

    .. warning: The transmission coefficients have a meaning only if the lower medium
    is lossless, or they have no true meaning.

    """
    # The medium may be dispersive. The permittivity and permability of each
    # layer has to be computed each time.
    Epsilon, Mu = struct.polarizability(wavelength)

    thickness = copy.deepcopy(struct.thickness)
    # In order to ensure that the phase reference is at the beginning
    # of the first layer.
    thickness[0] = 0
    Type = struct.layer_type
    # The boundary conditions will change when the polarization changes.
    if polarization == 0:
        f = Mu
    else:
        f = Epsilon
    # Wavevector in vacuum.
    k0 = 2 * np.pi / wavelength
    # Number of layers
    g = len(struct.layer_type)
    # Wavevector k_x, horizontal
    alpha = np.sqrt(Epsilon[Type[0]] * Mu[Type[0]]) * k0 * np.sin(incidence)
    # Computation of the vertical wavevectors k_z
    gamma = np.sqrt(
        Epsilon[Type] * Mu[Type] * k0 ** 2 - np.ones(g) * alpha ** 2)
    # Be cautious if the upper medium is a negative index one.
    if np.real(Epsilon[Type[0]]) < 0 and np.real(Mu[Type[0]]) < 0:
        gamma[0] = -gamma[0]

    # Changing the determination of the square root to achieve perfect stability
    if g > 2:
        gamma[1:g - 2] = gamma[1:g - 2] * (
                1 - 2 * (np.imag(gamma[1:g - 2]) < 0))
    # Outgoing wave condition for the last medium
    if np.real(Epsilon[Type[g - 1]]) < 0 and np.real(
            Mu[Type[g - 1]]) < 0 and np.real(np.sqrt(Epsilon[Type[g - 1]] * Mu[
        Type[g - 1]] * k0 ** 2 - alpha ** 2)) != 0:
        gamma[g - 1] = -np.sqrt(
            Epsilon[Type[g - 1]] * Mu[Type[g - 1]] * k0 ** 2 - alpha ** 2)
    else:
        gamma[g - 1] = np.sqrt(
            Epsilon[Type[g - 1]] * Mu[Type[g - 1]] * k0 ** 2 - alpha ** 2)
    T = np.zeros(((2 * g, 2, 2)), dtype=complex)

    # first S matrix
    T[0] = [[0, 1], [1, 0]]
    for k in range(g - 1):
        # Layer scattering matrix
        t = np.exp((1j) * gamma[k] * thickness[k])
        T[2 * k + 1] = [[0, t], [t, 0]]
        # Interface scattering matrix
        b1 = gamma[k] / f[Type[k]]
        b2 = gamma[k + 1] / f[Type[k + 1]]
        T[2 * k + 2] = np.array(
            [[b1 - b2, 2 * b2], [2 * b1, b2 - b1]] / (b1 + b2))
    t = np.exp((1j) * gamma[g - 1] * thickness[g - 1])
    T[2 * g - 1] = [[0, t], [t, 0]]
    # Once the scattering matrixes have been prepared, now let us combine them
    H = np.zeros(((2 * g - 1, 2, 2)), dtype=complex)
    A = np.zeros(((2 * g - 1, 2, 2)), dtype=complex)
    H[0] = T[2 * g - 1]
    A[0] = T[0]
    for k in range(len(T) - 2):
        A[k + 1] = cascade(A[k], T[k + 1])
        H[k + 1] = cascade(T[2 * g - 2 - k], H[k])

    I = np.zeros(((2 * g, 2, 2)), dtype=complex)
    for k in range(len(T) - 1):
        I[k] = np.array(
            [[A[k][1, 0], A[k][1, 1] * H[len(T) - k - 2][0, 1]],
             [A[k][1, 0] * H[len(T) - k - 2][0, 0],
              H[len(T) - k - 2][0, 1]]] / (
                    1 - A[k][1, 1] * H[len(T) - k - 2][0, 0]))
#        I[k][0, 0] = A[k][1, 0] / (1 - A[k][1, 1] * H[len(T) - 2 - k][0, 0])
#        I[k][0, 1] = A[k][1, 1] * H[len(T) - 2 - k][0, 1] / (
#                1 - A[k][1, 1] * H[len(T) - 2 - k][0, 0])
#        I[k][1, 0] = A[k][1, 0] * H[len(T) - 2 - k][0, 0] / (
#                1 - A[k][1, 1] * H[len(T) - 2 - k][0, 0])
#        I[k][1, 1] = H[len(T) - 2 - k][0, 1] / (
#                1 - A[k][1, 1] * H[len(T) - 2 - k][0, 0])
    I[2 * g - 1][0, 0] = I[2 * g - 2][0, 0] * np.exp(
        1j * gamma[g - 1] * thickness[g - 1])
    I[2 * g - 1][0, 1] = I[2 * g - 2][0, 1] * np.exp(
        1j * gamma[g - 1] * thickness[g - 1])
    I[2 * g - 1][1, 0] = 0
    I[2 * g - 1][1, 1] = 0

    w = 0
    poynting = np.zeros(2 * g, dtype=complex)
    if polarization == 0:  # TE
        for k in range(2 * g):
            poynting[k] = np.real((I[k][0, 0] + I[k][1, 0]) * np.conj(
                (I[k][0, 0] - I[k][1, 0]) * gamma[w] / Mu[Type[w]])) * Mu[
                              Type[0]] / (gamma[0])
            w = w + 1 - np.mod(k + 1, 2)
    else:  # TM
        for k in range(2 * g):
            poynting[k] = np.real((I[k][0, 0] - I[k][1, 0]) * np.conj(
                (I[k][0, 0] + I[k][1, 0]) * gamma[w] / Epsilon[Type[w]]) *
                                  Epsilon[Type[0]] / (gamma[0]))
            w = w + 1 - np.mod(k + 1, 2)
    # Absorption in each layer
    tmp = abs(-np.diff(poynting))
    # absorb=np.zeros(g,dtype=complex)
    absorb = tmp[np.arange(0, 2 * g, 2)]
    # reflection coefficient of the whole structure
    r = A[len(A) - 1][0, 0]
    # transmission coefficient of the whole structure
    t = A[len(A) - 1][1, 0]
    # Energy reflexion coefficient;
    R = np.real(abs(r) ** 2)
    # Energy transmission coefficient;
    T = np.real(
        abs(t) ** 2 * gamma[g - 1] * f[Type[0]] / (gamma[0] * f[Type[g - 1]]))

    return absorb, r, t, R, T


def field(struct, beam, window):
    """Computes the electric (TE polarization) or magnetic (TM) field inside
    a multilayered structure illuminated by a gaussian beam.

    Args:
        struct (Structure): description (materials,thicknesses)of the multilayer
        beam (Beam): description of the incidence beam
        window (Window): description of the simulation domain

    Returns:
        En (np.array): a matrix with the complex amplitude of the field

    Afterwards the matrix may be used to represent either the modulus or the
    real part of the field.
    """

    # Wavelength in vacuum.
    lam = beam.wavelength
    # Computation of all the permittivities/permeabilities
    Epsilon, Mu = struct.polarizability(lam)
    thickness = np.array(struct.thickness)
    w = beam.waist
    pol = beam.polarization
    d = window.width
    theta = beam.incidence
    C = window.C
    ny = np.floor(thickness / window.py)
    nx = window.nx
    Type = struct.layer_type
    print("Pixels vertically:", int(sum(ny)))

    # Number of modes retained for the description of the field
    # so that the last mode has an amplitude < 1e-3 - you may want
    # to change it if the structure present reflexion coefficients
    # that are subject to very swift changes with the angle of incidence.

    nmod = int(np.floor(0.83660 * d / w))

    # ----------- Do not touch this part ---------------
    l = lam / d
    w = w / d
    thickness = thickness / d

    if pol == 0:
        f = Mu
    else:
        f = Epsilon
    # Wavevector in vacuum, no dimension
    k0 = 2 * pi / l
    # Initialization of the field component
    En = np.zeros((int(sum(ny)), int(nx)))
    # Total number of layers
    # g=Type.size-1
    g = len(struct.layer_type) - 1
    # Amplitude of the different modes
    nmodvect = np.arange(-nmod, nmod + 1)
    # First factor makes the gaussian beam, the second one the shift
    # a constant phase is missing, it's just a change in the time origin.
    X = np.exp(-w ** 2 * pi ** 2 * nmodvect ** 2) * np.exp(
        -2 * 1j * pi * nmodvect * C)

    # Scattering matrix corresponding to no interface.
    T = np.zeros((2 * g + 2, 2, 2), dtype=complex)
    T[0] = [[0, 1], [1, 0]]
    for nm in np.arange(2 * nmod + 1):

        alpha = np.sqrt(Epsilon[Type[0]] * Mu[Type[0]]) * k0 * sin(
            theta) + 2 * pi * (nm - nmod)
        gamma = np.sqrt(
            Epsilon[Type] * Mu[Type] * k0 ** 2 - np.ones(g + 1) * alpha ** 2)

        if np.real(Epsilon[Type[0]]) < 0 and np.real(Mu[Type[0]]) < 0:
            gamma[0] = -gamma[0]

        if g > 2:
            gamma[1:g - 1] = gamma[1:g - 1] * (
                    1 - 2 * (np.imag(gamma[1:g - 1]) < 0))
        if np.real(Epsilon[Type[g]]) < 0 and np.real(
                Mu[Type[g]]) < 0 and np.real(
            np.sqrt(Epsilon[Type[g]] * k0 ** 2 - alpha ** 2)) != 0:
            gamma[g] = -np.sqrt(
                Epsilon[Type[g]] * Mu[Type[g]] * k0 ** 2 - alpha ** 2)
        else:
            gamma[g] = np.sqrt(
                Epsilon[Type[g]] * Mu[Type[g]] * k0 ** 2 - alpha ** 2)

        for k in range(g):
            t = np.exp(1j * gamma[k] * thickness[k])
            T[2 * k + 1] = np.array([[0, t], [t, 0]])
            b1 = gamma[k] / f[Type[k]]
            b2 = gamma[k + 1] / f[Type[k + 1]]
            T[2 * k + 2] = np.array([[b1 - b2, 2 * b2], [2 * b1, b2 - b1]]) / (
                    b1 + b2)
        t = np.exp(1j * gamma[g] * thickness[g])
        T[2 * g + 1] = np.array([[0, t], [t, 0]])

        H = np.zeros((len(T) - 1, 2, 2), dtype=complex)
        A = np.zeros((len(T) - 1, 2, 2), dtype=complex)

        H[0] = T[2 * g + 1]
        A[0] = T[0]

        for k in range(len(T) - 2):
            A[k + 1] = cascade(A[k], T[k + 1])
            H[k + 1] = cascade(T[len(T) - k - 2], H[k])

        I = np.zeros((len(T), 2, 2), dtype=complex)
        for k in range(len(T) - 1):
            I[k] = np.array(
                [[A[k][1, 0], A[k][1, 1] * H[len(T) - k - 2][0, 1]],
                 [A[k][1, 0] * H[len(T) - k - 2][0, 0],
                  H[len(T) - k - 2][0, 1]]] / (
                        1 - A[k][1, 1] * H[len(T) - k - 2][0, 0]))

        h = 0
        t = 0



        E = np.zeros((int(np.sum(ny)), 1), dtype=complex)
        for k in range(g + 1):
            for m in range(int(ny[k])):
                h = h + float(thickness[k]) / ny[k]
                #The expression for the field used here is based on the assumption
                # that the structure is illuminated from above only, with an Amplitude
                # of 1 for the incident wave. If you want only the reflected
                # field, take off the second term.
                E[t, 0] = I[2 * k][0, 0] * np.exp(1j * gamma[k] * h) + \
                          I[2 * k + 1][1, 0] * np.exp(
                    1j * gamma[k] * (thickness[k] - h))
                t += 1
            h = 0
        E = E * np.exp(1j * alpha * np.arange(0, nx) / nx)
        En = En + X[int(nm)] * E

    return En


def Angular(structure, wavelength, polarization, theta_min, theta_max,
            n_points):
    """Represents the reflexion coefficient (reflectance and phase) for a
    multilayered structure. This is an automated call to the :coefficient:
    function making the angle of incidence vary.

    Args:
        structure (Structure): the object describing the multilayer
        wavelength (float): the working wavelength in nm
        polarization (float): 0 for TE, 1 for TM
        theta_min (float): minimum angle of incidence in degrees
        theta_max (float): maximum angle of incidence in degrees
        n_points (int): number of different angle of incidence

    Returns:
        incidence (numpy array): angles of incidence considered
        r (numpy complex array): reflexion coefficient for each angle
        t (numpy complex array): transmission coefficient
        R (numpy array): Reflectance
        T (numpy array): Transmittance

    .. warning: The incidence angle is in degrees here, contrarily to
    other functions.

    """

    # theta min and max in degrees this time !
    import matplotlib.pyplot as plt
    r = np.zeros(n_points, dtype=complex)
    t = np.zeros(n_points, dtype=complex)
    R = np.zeros(n_points)
    T = np.zeros(n_points)
    incidence = np.zeros(n_points)
    incidence = np.linspace(theta_min, theta_max, n_points)
    for k in range(n_points):
        r[k], t[k], R[k], T[k] = coefficient(structure, wavelength,
                                             incidence[k] / 180 * np.pi,
                                             polarization)

    return incidence, r, t, R, T


def Spectrum(structure, incidence, polarization, wl_min, wl_max, n_points):
    """Represents the reflexion coefficient (reflectance and phase) for a
    multilayered structure. This is an automated call to the :coefficient:
    function making the angle of incidence vary.

    Args:
        structure (Structure): the object describing the multilayer
        incidence (float): incidence angle in degrees
        polarization (float): 0 for TE, 1 for TM
        wl_min (float): minimum wavelength of the spectrum
        theta_max (float): maximum wavelength of the spectrum
        n_points (int): number of points in the spectrum

    Returns:
        wl (numpy array): wavelength considered
        r (numpy complex array): reflexion coefficient for each wavelength
        t (numpy complex array): transmission coefficient
        R (numpy array): Reflectance
        T (numpy array): Transmittance


    .. warning: The incidence angle is in degrees here, contrarily to
    other functions.

    """
    # incidence in degrees
    import matplotlib.pyplot as plt
    r = np.zeros(n_points, dtype=complex)
    t = np.zeros(n_points, dtype=complex)
    R = np.zeros(n_points)
    T = np.zeros(n_points)
    wl = np.linspace(wl_min, wl_max, n_points)
    theta = incidence / 180 * np.pi
    for k in range(n_points):
        r[k], t[k], R[k], T[k] = coefficient(structure, wl[k], theta,
                                             polarization)
    return wl, r, t, R, T
