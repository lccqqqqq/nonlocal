# %% Testing for MPS normalizations

# This time, use real symmetric matrices for matrix product states. They have much nicer spectra!!!


import torch as t
import einops

# Constants
device = t.device("cuda" if t.cuda.is_available() else "cpu")

# Initialize a state
d_bond = 16
d_phys = 2
A = t.randn(d_bond, d_phys, d_bond, dtype=t.float64, device=device)
A = A + A.transpose(0, 2)


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
    pauli_z = t.tensor([[1, 0], [0, -1]], device=device, dtype=t.float64)
    pauli_x = t.tensor([[0, 1], [1, 0]], device=device, dtype=t.float64)
    h_loc = (
        -t.kron(pauli_z, pauli_z)
        - 1 / 2 * g * t.kron(pauli_x, t.eye(d_phys, device=device, dtype=t.float64))
        - 1 / 2 * g * t.kron(t.eye(d_phys, device=device, dtype=t.float64), pauli_x)
    )
    return h_loc



# %% Get the transfer matrix
eps = 1e-10

transfer_matrix = einops.einsum(
    A,
    A,
    "bl p br, blc p brc -> bl blc br brc",
)
transfer_matrix = einops.rearrange(
    transfer_matrix, "bl blc br brc -> (bl blc) (br brc)"
)


assert transfer_matrix.shape == (d_bond**2, d_bond**2)


# Normalizing the state is equivalent to finding the trace of the transfer matrix
n_site = 8
U, S, Vt = t.linalg.svd(transfer_matrix, full_matrices=True)

# Use eigensolver to find the dominant eigenvalue and eigenvector
eigenvalues, eigenvectors = t.linalg.eigh(transfer_matrix, UPLO='U')
# Get the index of the largest eigenvalue

pre_normalization = t.trace(transfer_matrix) + 1e-10
# pre-normalize so that the number does not overflow
trace_1 = t.trace(t.linalg.matrix_power(transfer_matrix/pre_normalization, n_site))
print(f"trace_1: {trace_1}")
trace_2 = t.trace(t.diag(eigenvalues/pre_normalization).pow(n_site))
print(f"trace_2: {trace_2}")
# assert t.allclose(trace_1, trace_2)
assert abs(trace_1 - trace_2) < 1e-10
# The matrices U and V are unitary
assert t.allclose(U @ U.H, t.eye(d_bond**2, device=device, dtype=t.float64))
assert t.allclose(Vt @ Vt.H, t.eye(d_bond**2, device=device, dtype=t.float64))


# %% Energy computation

def compute_transfer_matrix_and_prenormalize_state(A, eps=1e-10):
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
    A_prenormalized = A / t.sqrt(t.max(eigenvalues))
    # Here, we use the max eigenvalue as a proxy
    
    return A_prenormalized, transfer_matrix, eigval_prenormalized, eigenvectors, max_eigenvector

def compute_energy_density(A, g, n_site=None, eps=1e-10, return_state=False):
    A_prenormalized, transfer_matrix, eigval_prenormalized, eigenvectors, max_eigenvector = compute_transfer_matrix_and_prenormalize_state(A, eps)
    # compute the local energy term
    A_prod = einops.einsum(
        A_prenormalized,
        A_prenormalized,
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
        # 2. the fixed point is the max eigenvector, from both sides
        left_fixed_point = max_eigenvector.unsqueeze(0)
        right_fixed_point = max_eigenvector.unsqueeze(1)
        local_energy = t.real(
            left_fixed_point @ local_energy_op @ right_fixed_point
        ).squeeze()
        if return_state:
            A_normalized = A_prenormalized / t.sqrt(t.max(eigenvalues))
    else:
        # This is for finite chain, where we need to trace over the transfer matrix
        local_energy = t.tensor(0.0, device=device, dtype=t.float64)
        for i in range(len(eigval_prenormalized)):
            local_energy += eigval_prenormalized[i]**(n_site-2) * t.real(
                eigenvectors[:, i].unsqueeze(0) @ local_energy_op @ eigenvectors[:, i].unsqueeze(1)
            ).squeeze()
        
        exact_normalization = t.sum(eigval_prenormalized.pow(n_site))
        local_energy = local_energy / exact_normalization
        if return_state:
            A_normalized = A_prenormalized / exact_normalization**(1/(2*n_site))
    
    if return_state:
        return local_energy, A_normalized
    else:
        return local_energy

# %% Test functionality
energy_density, A_normalized = compute_energy_density(A, g=0.5, n_site=8, return_state=True)
print(energy_density)

# Assert that the state is normalized
test_transfer_matrix = einops.einsum(
    A_normalized,
    A_normalized,
    "bl p br, blc p brc -> bl blc br brc",
)
test_transfer_matrix = einops.rearrange(
    test_transfer_matrix, "bl blc br brc -> (bl blc) (br brc)"
)
trace = t.trace(t.linalg.matrix_power(test_transfer_matrix, 8))

# %% Test finite size effect

from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn

params = nn.Parameter(t.randn(d_phys, int(d_bond * (d_bond+1) / 2), dtype=t.float64, device=device))
optimizer = optim.Adam([params], lr=0.001)
pbar = tqdm(range(1000))
energy_density_history = []

def construct_umps(params):
    # Construct symmetric matrices from parameters
    # params has shape (d_phys, d_bond * (d_bond + 1) / 2)
    # We need to construct d_phys symmetric matrices of size d_bond x d_bond
    
    # Initialize output tensor
    matrices = t.zeros(d_phys, d_bond, d_bond, dtype=t.float64, device=device)
    
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

for i in pbar:
    optimizer.zero_grad()
    A = construct_umps(params)
    energy_density = compute_energy_density(A, g=0.5, n_site=8)
    energy_density.backward()
    optimizer.step()
    energy_density_history.append(energy_density.item())

    if i % 20 == 0:
        pbar.set_description(f"Energy density: {energy_density.item()}")


# %% We know the exact ground state in computational basis, so good to be able to verify that as well. Fix the amplitudes to real as well.

# 1. check ground state energy
# 2. check ground state norm
import plotly.express as px
px.line(energy_density_history, title="Energy density history")

# %%

# get the ground state energy
import math
import numpy as np
g = 0.5
N = 8
k_set = [-(N-1)/N * math.pi + 2*i/N * math.pi for i in range(N)]
eps = lambda k: math.sqrt(1 + g**2 - 2*g*math.cos(k))

E_gs = - sum([eps(k) for k in k_set]) / N

px.line(np.array(energy_density_history)-E_gs, title="Energy density history", log_x=False, log_y=True)


# %%
import torch as t
import numpy as np
from spin_model import h_ising_model

H_ising = h_ising_model(J=1.0, Gm=0.5, h=0.0, N=8, bc='PBC', flg=0)
print(H_ising)

eigval, eigvec = np.linalg.eigh(H_ising)
# print(eigval/8)
# print(eigvec[0])

# 

# %% Testing the exact ground state
# we decide to use the exact diagonalization result instead of resorting to the analytical expressions

from quspin.operators import hamiltonian  # Hamiltonians and operators
from quspin.basis import (
    spin_basis_1d,
    spinless_fermion_basis_1d,
)  # Hilbert space spin basis
import numpy as np  # generic math functions
import matplotlib.pyplot as plt  # plotting library

#
##### define model parameters #####
L = 8  # system size
J = 1.0  # spin zz interaction
h = 0.4  # z magnetic field strength
#
# loop over spin inversion symmetry block variable and boundary conditions
for zblock, PBC in zip([1], [1]):
    #
    ##### define spin model
    # site-coupling lists (PBC for both spin inversion sectors)
    h_field = [[-h, i] for i in range(L)]
    J_zz = [[-J, i, (i + 1) % L] for i in range(L)]  # PBC
    # define spin static and dynamic lists
    static_spin = [["zz", J_zz], ["x", h_field]]  # static part of H
    dynamic_spin = []  # time-dependent part of H
    # construct spin basis in pos/neg spin inversion sector depending on APBC/PBC
    basis_spin = spin_basis_1d(L=L, zblock=zblock)
    # build spin Hamiltonians
    H_spin = hamiltonian(static_spin, dynamic_spin, basis=basis_spin, dtype=np.float64)
    # calculate spin energy levels
    E_spin = H_spin.eigvalsh()
    #
    ##### define fermion model
    # define site-coupling lists for external field
    h_pot = [[2.0 * h, i] for i in range(L)]
    if PBC == 1:  # periodic BC: odd particle number subspace only
        # define site-coupling lists (including boudary couplings)
        J_pm = [[-J, i, (i + 1) % L] for i in range(L)]  # PBC
        J_mp = [[+J, i, (i + 1) % L] for i in range(L)]  # PBC
        J_pp = [[-J, i, (i + 1) % L] for i in range(L)]  # PBC
        J_mm = [[+J, i, (i + 1) % L] for i in range(L)]  # PBC
        # construct fermion basis in the odd particle number subsector
        basis_fermion = spinless_fermion_basis_1d(L=L, Nf=range(1, L + 1, 2))
    elif PBC == -1:  # anti-periodic BC: even particle number subspace only
        # define bulk site coupling lists
        J_pm = [[-J, i, i + 1] for i in range(L - 1)]
        J_mp = [[+J, i, i + 1] for i in range(L - 1)]
        J_pp = [[-J, i, i + 1] for i in range(L - 1)]
        J_mm = [[+J, i, i + 1] for i in range(L - 1)]
        # add boundary coupling between sites (L-1,0)
        J_pm.append([+J, L - 1, 0])  # APBC
        J_mp.append([-J, L - 1, 0])  # APBC
        J_pp.append([+J, L - 1, 0])  # APBC
        J_mm.append([-J, L - 1, 0])  # APBC
        # construct fermion basis in the even particle number subsector
        basis_fermion = spinless_fermion_basis_1d(L=L, Nf=range(0, L + 1, 2))
    # define fermionic static and dynamic lists
    static_fermion = [
        ["+-", J_pm],
        ["-+", J_mp],
        ["++", J_pp],
        ["--", J_mm],
        ["z", h_pot],
    ]
    dynamic_fermion = []
    # build fermionic Hamiltonian
    H_fermion = hamiltonian(
        static_fermion,
        dynamic_fermion,
        basis=basis_fermion,
        dtype=np.float64,
        check_pcon=False,
        check_symm=False,
    )
    # calculate fermionic energy levels
    E_fermion = H_fermion.eigvalsh()
    #
    ##### plot spectra
    plt.plot(
        np.arange(H_fermion.Ns), E_fermion / L, marker="o", color="b", label="fermion"
    )
    plt.plot(
        np.arange(H_spin.Ns),
        E_spin / L,
        marker="x",
        color="r",
        markersize=2,
        label="spin",
    )
    plt.xlabel("state number", fontsize=16)
    plt.ylabel("energy", fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)
    plt.grid()
    plt.tight_layout()
    plt.savefig("example4.pdf", bbox_inches="tight")
    plt.show()
    # plt.close()








