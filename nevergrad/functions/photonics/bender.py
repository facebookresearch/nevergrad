# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as lin

# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
from .functions import addupml2d, yeeder2d, block

# import time

# %% CALCULATE S-PARAMETERS FOR WGs


c0 = 299792458
epsi0 = 8.85418782e-12
mu0 = 12.566370614e-7
Z0 = np.sqrt(mu0 / epsi0)

# UNITS in micrometers
micrometers = 1
nanometers = micrometers / 1000

#########################################################
## Using Ceviche mode converter dimentions
#########################################################


def bender(X, ev_out=-(3.5**2), ev_in=-(3.5**2)):
    """
    Computes the conversion efficiency between the mode of index ev_in
    in the input wg, into the mode of index ev_out in the output wg,
    given a central block of dimensions:
        centerWG_w = 2 * micrometers        # Width
        centerWG_L = 2 * micrometers          # Length
    described in X float array of size (centerWG_w x dx , centerWG_L x dy)
    X is the permittivities
    Default ev values compute fundamental mode to first order mode
    """

    # SOURCE
    lam0 = 1.55 * micrometers
    k0 = 2 * np.pi / lam0

    MODE = "E"  # 'E' or 'H'

    # CENTER modifiable box PARAMETERS
    centerWG_n = 3.5  # refractive index, if discretized
    centerWG_w = 1.5 * micrometers  # Width
    centerWG_L = 1.5 * micrometers  # Length

    # INPUT / OUTPUT WGs  PARAMETERS
    inputWG_n = 3.5  # refractive index
    inputWG_L = 750 * nanometers  # Width
    inputWG_w = 200 * nanometers  # Length
    shift_in = 00 * nanometers  # y-shift

    outputWG_n = 3.5  # refractive index
    outputWG_L = 750 * nanometers  # Width
    outputWG_w = 200 * nanometers  # Length
    shift_out = -00 * nanometers  # x-shift

    # FDFD PARAMETERS
    NRES = 5  # GRID RESOLUTION
    SPACER = lam0 * np.array([1, 1])  # X and Y SPACERS
    NPML = [20, 20, 20, 20]  # NB PML
    nmax = max([inputWG_n, centerWG_n, outputWG_n])

    ### CALCULATE OPTIMIZED GRID
    # GRID RESOLUTION
    dx = lam0 / (nmax * NRES)
    dy = lam0 / (nmax * NRES)

    # SNAP GRID TO CRITICAL DIMENSIONS
    a = np.min([inputWG_w, centerWG_w, outputWG_w])
    nx = np.ceil(a / dx)
    dx = a / nx
    ny = np.ceil(a / dy)
    dy = a / ny

    # GRID SIZE
    grid_size_x = inputWG_L + centerWG_L + SPACER[1]
    Nx = int(NPML[0] + np.ceil(grid_size_x / dx) + NPML[1])
    grid_size_x = Nx * dx

    grid_size_y = SPACER[0] + centerWG_w + outputWG_L
    Ny = int(NPML[2] + np.ceil(grid_size_y / dy) + NPML[3])
    grid_size_y = Ny * dy

    # 2X GRID
    Nx2 = 2 * Nx
    dx2 = dx / 2
    Ny2 = 2 * Ny
    dy2 = dy / 2

    # CALCULATE AXIS VECTORS
    # xa = np.arange(1, Nx+1) * dx
    # ya = np.arange(1, Ny+1) * dy
    xa2 = np.arange(1, Nx2 + 1) * dx2
    ya2 = np.arange(1, Ny2 + 1) * dy2
    # # CENTER WINDOW (with (x,y) = (0,0) in the center of the central block)
    # xa = xa - np.mean(xa)
    # ya = ya - np.mean(ya)
    # xa2 = xa2 - np.mean(xa2)
    # ya2 = ya2 -np.mean(ya2)

    PML = [np.abs(xa2[2 * NPML[0] - 1] - xa2[0]), np.abs(ya2[Ny2 - 1] - ya2[Ny2 - 1 - 2 * NPML[3]])]

    # %% BUILD OPTICAL INTEGRATED CIRCUIT
    # INITIALIZE MATERIALS ARRAYS
    ER2 = np.ones((Nx2, Ny2))
    UR2 = np.ones((Nx2, Ny2))

    # INPUT WAVEGUIDE
    pos_x = 0
    pos_y = PML[1] + SPACER[0] + centerWG_w / 2
    Len_x = PML[0] + inputWG_L
    Len_y = inputWG_w
    pos_y = pos_y + shift_in
    ER2 = block(xa2, ya2, ER2, pos_x, pos_y, Len_x, Len_y, inputWG_n)

    # OUTPUT WAVEGUIDE
    pos_x = PML[0] + inputWG_L + centerWG_L / 2
    pos_y = PML[1] + SPACER[0] + centerWG_w
    Len_x = outputWG_w
    Len_y = outputWG_L + PML[1]
    pos_x = pos_x + shift_out
    ER2 = block(xa2, ya2, ER2, pos_x, pos_y, Len_x, Len_y, outputWG_n)

    nb_struct_x = int(centerWG_L / dx2)
    nb_struct_y = int(centerWG_w / dy2)
    assert (
        len(X) == nb_struct_x and len(X[0]) == nb_struct_y
    ), f"nb_struct_x={nb_struct_x}, nb_struct_y={nb_struct_y}, np.shape(X)={np.shape(X)}"

    #  CENTER WAVEGUIDE
    X = 1 + X * (centerWG_n**2 - 1)  # shift value range from 0-1 to 1-n**2

    center_x_beg = int((inputWG_L + PML[0]) / dx2)
    center_y_beg = int((SPACER[0] + PML[1]) / dx2)

    ER2[center_x_beg : center_x_beg + nb_struct_x, center_y_beg : center_y_beg + nb_struct_y] = X

    # INCORPORATE PML
    [inv_ERxx, inv_ERyy, ERzz, inv_URxx, inv_URyy, URzz] = addupml2d(ER2, UR2, NPML)

    #########################################################
    ## ANALYZE INPUT / OUTPUT WAVEGUIDES
    #########################################################

    # EXTRACT INPUT SLAB WAVEGUIDE FROM GRID
    nx = 2 * NPML[0] + 1
    # Input mode starts after the PML, so we can retrieve the
    # permittivities directly
    inv_erxx = sp.diags_array(1 / ER2[nx, 0:Ny2:2], format="csc")
    inv_eryy = sp.diags_array(1 / ER2[nx, 1:Ny2:2], format="csc")
    erzz = sp.diags_array(ER2[nx, 0:Ny2:2], format="csc")
    inv_urxx = sp.diags_array(1 / UR2[nx, 1:Ny2:2], format="csc")
    inv_uryy = sp.diags_array(1 / UR2[nx + 1, 0:Ny2:2], format="csc")
    urzz = sp.diags_array(UR2[nx, 1:Ny2:2], format="csc")

    # BUILD DERIVATIVE MATRICES
    NS = [1, Ny]
    RES = np.array([1, dy])
    BC = [0, 0]
    [_, DEY, _, DHY] = yeeder2d(NS, k0 * RES, BC)

    # BUILD EIGEN-VALUE PROBLEM
    if MODE == "E":
        A = -(DHY @ inv_urxx @ DEY + erzz)
        B = inv_uryy
    else:
        A = -(DEY @ inv_erxx @ DHY + urzz)
        B = inv_eryy

    # SOLVE FULL EIGEN-VALUE PROBLEM
    ev = ev_in
    [D, input_mode] = lin.eigs(A, k=1, M=B, sigma=ev)  # Find the eigen value closest to the -n**2
    D = np.sqrt(D)
    NEFF = -1j * D
    # GET SOURCE MODE
    neff_in = NEFF[0]
    normalization = np.sqrt(neff_in / (2 * Z0) * (np.sum(np.abs(input_mode[:, 0]) ** 2) * dy))
    input_mode = input_mode[:, 0] / normalization  # NORMALIZED MODE

    # EXTRACT OUPUT SLAB WAVEGUIDE FROM GRID
    ny = Ny2 - 2 * NPML[3] - 1
    # Output mode is before the PML, so we can retrieve the
    # permittivities directly
    erzz = sp.diags_array(ER2[0:Nx2:2, ny], format="csc")
    inv_erxx = sp.diags_array(1 / ER2[0:Nx2:2, ny], format="csc")
    inv_eryy = sp.diags_array(1 / ER2[1:Nx2:2, ny], format="csc")
    inv_urxx = sp.diags_array(1 / UR2[1:Nx2:2, ny], format="csc")
    inv_uryy = sp.diags_array(1 / UR2[0:Nx2:2, ny + 1], format="csc")
    urzz = sp.diags_array(UR2[1:Nx2:2, ny], format="csc")

    # BUILD DERIVATIVE MATRICES
    NS = [Nx, 1]
    RES = np.array([dx, 1])
    BC = [0, 0]
    # Building derivatives along an x-axis
    [DEX, _, DHX, _] = yeeder2d(NS, k0 * RES, BC)

    # BUILD EIGEN-VALUE PROBLEM
    if MODE == "E":
        A = -(DHX @ inv_uryy @ DEX + erzz)
        B = inv_urxx
    else:
        A = -(DEX @ inv_eryy @ DHX + urzz)
        B = inv_erxx

    # SOLVE FULL EIGEN-VALUE PROBLEM
    ev = ev_out
    [D, output_mode] = lin.eigs(A, k=1, M=B, sigma=ev)  # Find the eigen value closest to the -n**2
    D = np.sqrt(D)
    NEFF = -1j * D
    # GET SOURCE MODE
    neff_out = NEFF[0]
    normalization = np.sqrt(neff_out / (2 * Z0) * (np.sum(np.abs(output_mode[:, 0]) ** 2) * dy))
    output_mode = output_mode[:, 0] / normalization

    #########################################################
    ## PERFORM FDFD ANALYSIS
    #########################################################

    # BUILD DERIVATIVE MATRICES
    NS = [Nx, Ny]
    RES = np.array([dx, dy])
    BC = [0, 0]
    [DEX, DEY, DHX, DHY] = yeeder2d(NS, k0 * RES, BC)

    # BUILD WAVE MATRIX
    if MODE == "E":
        A = DHX @ inv_URyy @ DEX + DHY @ inv_URxx @ DEY + ERzz
    else:
        A = DEX @ inv_ERyy @ DHX + DEY @ inv_ERxx @ DHY + URzz

    # np.savetxt("Debug", A.toarray()[:100,:100], fmt="%.3f")
    # CALCULATE SOURCE FIELD
    fsrc = np.zeros((Nx, Ny), dtype=complex)
    for nx in range(Nx):
        fsrc[nx, :] = input_mode * np.exp(-1j * k0 * neff_in * (nx + 1) * dx)

    # CALCULATE SCATTERED-FIELD MASKING MATRIX
    nx = NPML[0] + 2
    Q = np.zeros((Nx, Ny))
    Q[:nx, :] = 1
    Q = sp.diags_array(Q.flatten("F"), format="csc")

    # CALCULATE SOURCE VECTOR
    b = (Q @ A - A @ Q) @ fsrc.flatten("F")
    # SOLVE FOR FIELD
    f = lin.spsolve(A, b)
    U = np.reshape(f, (Nx, Ny), order="F")

    #########################################################
    ## ANALYZE TRANSMITTED AND REFLECTED
    #########################################################

    # EXTRACT FIELDS
    # nx = NPML[0]
    # fref = U[nx-1,:]
    ny = Ny - NPML[3] - 2
    ftrn = U[:, ny]

    # CALCULATE S PARAMETERS
    S21 = ftrn @ np.conj(output_mode) / (np.conj(output_mode).T @ output_mode)
    ST = abs(S21) ** 2  # TRANSMISSION
    return 1 - ST, U
