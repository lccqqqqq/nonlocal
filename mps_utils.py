import torch as t
import numpy as np
import einops
device = t.device('cuda' if t.cuda.is_available() else 'cpu')

def get_local_term(g: float, device=device):
    """Returns the local Hamiltonian term for the Transverse Field Ising Model (TFIM).
    
    The TFIM Hamiltonian is given by:
    H = -J Σᵢⱼ ZᵢZⱼ - h Σᵢ Xᵢ
    
    where:
    - J is the coupling strength (set to 1 in this implementation)
    - h is the transverse field strength (parameter g in this function)
    - Zᵢ and Xᵢ are Pauli matrices at site i
    
    This function returns the local term for two adjacent sites, which includes:
    - The ZZ interaction between the sites
    - The X field terms on each site
    
    Args:
        g (float): The transverse field strength parameter, treated J=1
        
    Returns:
        torch.Tensor: The local Hamiltonian term as a 4x4 matrix
    """
    d_phys = 2 # using local Hilbert space dimension = 2
    pauli_z = t.tensor([[1, 0], [0, -1]], device=device, dtype=t.float64)
    pauli_x = t.tensor([[0, 1], [1, 0]], device=device, dtype=t.float64)
    h_loc = (
        -t.kron(pauli_z, pauli_z)
        - 1 / 2 * g * t.kron(pauli_x, t.eye(d_phys, device=device, dtype=t.float64))
        - 1 / 2 * g * t.kron(t.eye(d_phys, device=device, dtype=t.float64), pauli_x)
    )
    return h_loc



def compute_transfer_matrix_and_prenormalize_state(A, eps=1e-10):
    """
    Compute the transfer matrix and prenormalize the state
    
    Transfer matrix is defined as the kronecker product of A and A.T=A in our use case.
    
    The prenormalized state avoid precision overflow, especially when moving towards thermodynamic limit. Also, this step normalizes the max eigenvalue of the transfer matrix to 1, and this is asymptotically exact for infinite systems.
    """
    assert t.allclose(A, A.transpose(0, 2)), "A must be real and symmetric along the bond dimensions"
    transfer_matrix = einops.einsum(
        A,
        A,
        "bl p br, blc p brc -> bl blc br brc",
    )
    transfer_matrix = einops.rearrange(
        transfer_matrix, "bl blc br brc -> (bl blc) (br brc)"
    )
    
    eigenvalues, eigenvectors = t.linalg.eigh(transfer_matrix, UPLO='U')
    assert t.max(eigenvalues) > eps, "Bad initialization or transfer matrix becomes negative definite"
    max_eigenvalue_idx = t.argmax(eigenvalues)
    eigval_prenormalized = eigenvalues / t.max(eigenvalues)
    max_eigenvector = eigenvectors[:, max_eigenvalue_idx]
    transfer_matrix = transfer_matrix / t.max(eigenvalues)
    A = A / t.sqrt(t.max(eigenvalues))  
    # Here, we use the max eigenvalue as a proxy
    
    return A, transfer_matrix, eigval_prenormalized, eigenvectors, max_eigenvector

def compute_energy_density(A, g, n_site=None, eps=1e-10, return_state=False):
    """
    Compute the energy density of the state
    """ 
    A, transfer_matrix, eigval_prenormalized, eigenvectors, max_eigenvector = compute_transfer_matrix_and_prenormalize_state(A, eps)
    # compute the local energy term
    A_prod = einops.einsum(
        A,
        A,
        "bl1 p1 b, b p2 br2 -> bl1 p1 p2 br2",
    ).flatten(1, 2)
    
    local_energy_op = einops.einsum(
        A_prod,
        get_local_term(g),
        A_prod,
        "bl phys br, phys physc, blc physc brc -> bl blc br brc",
    )
    local_energy_op = einops.rearrange(
        local_energy_op, "bl blc br brc -> (bl blc) (br brc)"
    )
    
    if n_site is None:
        # This is for infinite chain, where
        # 1. the prenormalization is exact
        # 2. the fixed point is the max genvector, from both sides
        left_fixed_point = max_eigenvector.unsqueeze(0)
        right_fixed_point = max_eigenvector.unsqueeze(1)
        local_energy = t.real(
            left_fixed_point @ local_energy_op @ right_fixed_point
        ).squeeze()

        A = A / t.sqrt(t.max(eigval_prenormalized))
    else:
        # This is for finite chain, where we need to trace over the transfer matrix
        local_energy = t.tensor(0.0, device=device, dtype=t.float64)
        for i in range(len(eigval_prenormalized)):
            local_energy += eigval_prenormalized[i]**(n_site-2) * t.real(
                eigenvectors[:, i].unsqueeze(0) @ local_energy_op @ eigenvectors[:, i].unsqueeze(1)
            ).squeeze()
        
        exact_normalization = t.sum(eigval_prenormalized.pow(n_site))
        local_energy = local_energy / exact_normalization

        A = A / exact_normalization**(1/(2*n_site))
    
    if return_state:
        return local_energy, A
    else:
        return local_energy


def construct_umps(params, d_bond):
    # Construct symmetric matrices from parameters
    # params has shape (d_phys, d_bond * (d_bond + 1) / 2)
    # We need to construct d_phys symmetric matrices of size d_bond x d_bond
    
    # Initialize output tensor
    assert params.shape[1] == int(d_bond * (d_bond + 1) / 2), "The parameter has inconsistent dimensions for the targeted d_bond. Note that we assume the matrix to be real and symmetric."
    assert params.shape[0] == 2, "Local Hilbert space dimension should be 2."
    matrices = t.zeros(2, d_bond, d_bond, dtype=t.float64, device=device)
    
    # Fill in the upper triangular part including diagonal
    idx = 0
    for i in range(d_bond):
        for j in range(i, d_bond):
            matrices[:, i, j] = params[:, idx]
            if i != j:  # If not on diagonal, also fill lower triangular part
                matrices[:, j, i] = params[:, idx]
            idx += 1
            
    assert t.allclose(matrices, matrices.transpose(1, 2)), "Matrices must be symmetric"
    return einops.rearrange(matrices, "p bl br -> bl p br")
