import numpy as np
from functools import reduce

def h_ising_model(J, Gm, h, N, bc='PBC', flg=0, return_local=False):
    """
    Create 2^N dimensional matrix representation of the global Hamiltonian of
    Ising model
    
    Parameters:
    -----------
    J : float
        Coupling constant for nearest-neighbor interactions
    Gm : float
        Transverse field strength (coefficient for X terms)
    h : float
        Longitudinal field strength (coefficient for Z terms)
    N : int
        Number of sites in the system
    bc : str
        Boundary conditions: 'PBC' (periodic) or 'OBC' (open)
    flg : int
        Flag for local Hamiltonian construction (0 for decomposition, 1 for feature minimization)
        
    Returns:
    --------
    H : numpy.ndarray
        Global Hamiltonian matrix (2^N x 2^N)
    Hloc : numpy.ndarray
        Local Hamiltonian matrices (2^N x 2^N x N)
    """
    # Initialize Hamiltonian matrices
    dim = 2**N
    H = np.zeros((dim, dim), dtype=float)
    Hloc = np.zeros((dim, dim, N), dtype=float)
    
    # Pauli matrices
    I = np.array([[1, 0], [0, 1]])
    X = np.array([[0, 1], [1, 0]])
    Z = np.array([[1, 0], [0, -1]])
    
    for i in range(N):
        if bc == 'PBC':  # Periodic boundary conditions
            jr = (i + 1) % N
            jl = (i - 1) % N
        elif bc == 'OBC':  # Open boundary conditions
            if i == N - 1:
                jr = -1  # Equivalent to MATLAB's 0 (no right neighbor)
                jl = N - 2
            elif i == 0:
                jl = -1  # Equivalent to MATLAB's 0 (no left neighbor)
                jr = 1
            else:
                jr = i + 1
                jl = i - 1
        
        # ZZ term for sites i, i+1
        if jr != -1:
            # Create tensor product for ZZ interaction
            operators = [I] * N
            operators[i] = Z
            operators[jr] = Z
            tmpr = reduce(np.kron, operators)
            
            H -= 0.5 * J * tmpr
        else:
            tmpr = np.zeros((dim, dim))
        
        if jl != -1:
            # Create tensor product for ZZ interaction
            operators = [I] * N
            operators[i] = Z
            operators[jl] = Z
            tmpl = reduce(np.kron, operators)
            
            H -= 0.5 * J * tmpl
        else:
            tmpl = np.zeros((dim, dim))
        
        # X term for site i
        operators = [I] * N
        operators[i] = X
        tmpx = reduce(np.kron, operators)
        H -= Gm * tmpx
        
        # Z term for site i
        operators = [I] * N
        operators[i] = Z
        tmpz = reduce(np.kron, operators)
        H -= h * tmpz
        
        if flg == 0:  # Decomposition of global Hamiltonian
            Hloc[:, :, i] = -0.5 * J * (tmpl + tmpr) - Gm * tmpx - h * tmpz
        elif flg == 1:  # Feature minimization of local energy
            Hloc[:, :, i] = -J * (tmpl + tmpr) - Gm * tmpx - h * tmpz
    
    if return_local:
        return H, Hloc
    else:
        return H

