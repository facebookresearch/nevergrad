# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import scipy.sparse as sp


def addupml2d(ER2, UR2, NPML):
    """
    ADDUPML2D     Add UPML to a 2D Yee Grid

    [ERxx,ERyy,ERzz,URxx,URyy,URzz] = addupml2d(ER2,UR2,NPML)

    INPUT ARGUMENTS
    ================
    ER2       Relative Permittivity on 2x Grid
    UR2       Relative Permeability on 2x Grid
    NPML      [NXLO NXHI NYLO NYHI] Size of UPML on 1x Grid

    OUTPUT ARGUMENTS
    ================
    ERxx      xx Tensor Element for Relative Permittivity
    ERyy      yy Tensor Element for Relative Permittivity
    ERzz      zz Tensor Element for Relative Permittivity
    URxx      xx Tensor Element for Relative Permeability
    URyy      yy Tensor Element for Relative Permeability
    URzz      zz Tensor Element for Relative Permeability
    """
    #########################################################
    ## INITIALIZE FUNCTION
    #########################################################

    # DEFINE PML PARAMETERS
    amax = 4
    cmax = 1
    p = 3

    # EXTRACT GRID PARAMETERS
    [Nx2, Ny2] = np.shape(ER2)

    # EXTRACT PML PARAMETERS
    NXLO = 2 * NPML[0]
    NXHI = 2 * NPML[1]
    NYLO = 2 * NPML[2]
    NYHI = 2 * NPML[3]

    #########################################################
    ## CALCULATE PML PARAMETERS
    #########################################################

    # INITIALIZE PML PARAMETERS TO PROBLEM SPACE
    sx = np.ones((Nx2, Ny2), dtype=complex)
    sy = np.ones((Nx2, Ny2), dtype=complex)

    # ADD XLO PML
    for nx in range(1, NXLO + 1):
        ax = 1 + (amax - 1) * (nx / NXLO) ** p
        cx = cmax * np.sin(0.5 * np.pi * nx / NXLO) ** 2
        sx[NXLO - nx, :] = ax * (1 - 1j * 60 * cx)

    # ADD XHI PML
    for nx in range(1, NXHI + 1):
        ax = 1 + (amax - 1) * (nx / NXHI) ** p
        cx = cmax * np.sin(0.5 * np.pi * nx / NXHI) ** 2
        sx[Nx2 - NXHI + nx - 1, :] = ax * (1 - 1j * 60 * cx)

    # ADD YLO PML
    for ny in range(1, NYLO + 1):
        ay = 1 + (amax - 1) * (ny / NYLO) ** p
        cy = cmax * np.sin(0.5 * np.pi * ny / NYLO) ** 2
        sy[:, NYLO - ny] = ay * (1 - 1j * 60 * cy)

    # ADD YHI PML
    for ny in range(1, NYHI + 1):
        ay = 1 + (amax - 1) * (ny / NYHI) ** p
        cy = cmax * np.sin(0.5 * np.pi * ny / NYHI) ** 2
        sy[:, Ny2 - NYHI + ny - 1] = ay * (1 - 1j * 60 * cy)

    #########################################################
    ## INCORPORATE PML
    #########################################################

    # CALCULATE TENSOR ELEMENTS WITH UPML
    ERxx = ER2 / sx * sy
    ERyy = ER2 * sx / sy
    ERzz = ER2 * sx * sy

    URxx = UR2 / sx * sy
    URyy = UR2 * sx / sy
    URzz = UR2 * sx * sy

    # EXTRACT TENSOR ELEMENTS ON YEE GRID
    ERxx = ERxx[1:Nx2:2, 0:Ny2:2]
    ERyy = ERyy[0:Nx2:2, 1:Ny2:2]
    ERzz = ERzz[0:Nx2:2, 0:Ny2:2]

    URxx = URxx[0:Nx2:2, 1:Ny2:2]
    URyy = URyy[1:Nx2:2, 0:Ny2:2]
    URzz = URzz[1:Nx2:2, 1:Ny2:2]

    # DIAGONALIZE MATERIAL TENSORS
    inv_ERxx = sp.diags_array(1 / ERxx.flatten("F"), format="csc")
    inv_ERyy = sp.diags_array(1 / ERyy.flatten("F"), format="csc")
    ERzz = sp.diags_array(ERzz.flatten("F"), format="csc")
    inv_URxx = sp.diags_array(1 / URxx.flatten("F"), format="csc")
    inv_URyy = sp.diags_array(1 / URyy.flatten("F"), format="csc")
    URzz = sp.diags_array(URzz.flatten("F"), format="csc")

    return [inv_ERxx, inv_ERyy, ERzz, inv_URxx, inv_URyy, URzz]


def yeeder2d(NS, RES, BC, kinc=0):
    """
    YEEDER2D      Derivative Matrices on a 2D Yee Grid

    [DEX,DEY,DHX,DHY] = yeeder2d(NS,RES,BC,kinc)

    INPUT ARGUMENTS
    =================
    NS    [Nx Ny] Grid Size
    RES   [dx dy] Grid Resolution
    BC    [xbc ybc] Boundary Conditions
            0: Dirichlet boundary conditions
            1: Periodic boundary conditions
    kinc  [kx ky] Incident Wave Vector
          This argument is only needed for PBCs.

    Note: For normalized grids use k0 * RES and kinc/k0

    OUTPUT ARGUMENTS
    =================
    DEX   Derivative Matrix wrt x for Electric Fields
    DEY   Derivative Matrix wrt to y for Electric Fields
    DHX   Derivative Matrix wrt to x for Magnetic Fields
    DHY   Derivative Matrix wrt to y for Magnetic Fields
    """

    #########################################################
    ## HANDLE INPUT ARGUMENTS
    #########################################################

    # EXTRACT GRID PARAMETERS
    Nx = NS[0]
    dx = RES[0]
    Ny = NS[1]
    dy = RES[1]

    # DEFAULT KINC
    if not (kinc):
        kinc = [0, 0]

    # DETERMINE MATRIX SIZE
    M = Nx * Ny

    # ZERO MATRIX
    Z = sp.csc_array((M, M))

    #########################################################
    ## BUILD DEX
    #########################################################

    # HANDLE IF SIZE IS 1 CELL
    if Nx == 1:
        DEX = -1j * kinc[0] * sp.eye_array(M)

    # HANDLE ALL OTHER CASES
    else:

        # Center Diagonal
        d0 = -np.ones(M)

        # Upper Diagonal
        d1 = np.ones(M - 1)
        d1[Nx - 1 : M : Nx] = 0

        # Build Derivative Matrix with Dirichlet BCs
        DEX = Z + sp.diags_array(d0 / dx)
        DEX = DEX + sp.diags_array(d1 / dx, offsets=1)

        # Incorporate Periodic Boundary Conditions
        if BC[0] == 1:
            d1 = sp.csc_array(M)
            d1[:M:Nx] = np.exp(-1j * kinc[0] * Nx * dx) / dx
            DEX = DEX + sp.diags_array(d1, offsets=1 - Nx)

    #########################################################
    ## BUILD DEY
    #########################################################

    # HANDLE IF SIZE IS 1 CELL
    if Ny == 1:
        DEY = -1j * kinc[1] * sp.eye_array(M)

    # HANDLE ALL OTHER CASES
    else:

        # Center Diagonal
        d0 = -np.ones((M))

        # Upper Diagonal
        d1 = np.ones((M - Nx))

        # Build Derivative Matrix with Dirichlet BCs
        DEY = Z + sp.diags_array(d0 / dy)
        DEY = DEY + sp.diags_array(d1 / dy, offsets=Nx)

        # Incorporate Periodic Boundary Conditions
        if BC[1] == 1:
            d1 = (np.exp(-1j * kinc[1] * Ny * dy) / dy) * np.ones((M, 1))
            DEY = DEY + sp.diags_array(d1, offsets=Nx - M)

    #########################################################
    ## BUILD DHX AND DHY
    #########################################################

    DHX = -np.conj(DEX.T)
    DHY = -np.conj(DEY.T)
    return [DEX, DEY, DHX, DHY]


def block(xa2, ya2, ER2, pos_x, pos_y, Len_x, Len_y, n):
    dx2 = xa2[1] - xa2[0]
    dy2 = ya2[1] - ya2[0]
    # MAKE n-index block whose corner left-down is at pos_x and pos_y
    # of size Len_x X Len_y
    # refractive index n

    ind_X = np.argmin(np.abs(xa2 - pos_x))
    ind_Y = np.argmin(np.abs(ya2 - pos_y))

    nx = int(ind_X + round(Len_x / dx2))
    ny = int(ind_Y + round(Len_y / dy2))

    ER2[ind_X:nx, ind_Y:ny] = n**2
    return ER2
