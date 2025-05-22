# %%
# finding the MPS representation of the Ising model ground states

import torch as t
import einops

# %% initialize a tensor

d_bond = 16
d_phys = 2
# need a real part and an imaginary part
# can wrap to nn.parameters
A_real = t.randn(d_bond, d_phys, d_bond, dtype=t.float64)
A_imag = t.randn(d_bond, d_phys, d_bond, dtype=t.float64)

# creating the Hamiltonian local term


def get_local_term(g: float):
    # using - ZZ - gX convention
    pauli_z = t.tensor([[1, 0], [0, -1]], dtype=t.complex128)
    pauli_x = t.tensor([[0, 1], [1, 0]], dtype=t.complex128)
    h_loc = (
        -t.kron(pauli_z, pauli_z)
        - 1 / 2 * g * t.kron(pauli_x, t.eye(d_phys))
        - 1 / 2 * g * t.kron(t.eye(d_phys), pauli_x)
    )
    return h_loc


# %% creating the central effective term for contraction

A = A_real + 1j * A_imag
A_prod = einops.einsum(
    A,
    A,
    "bl1 p1 b, b p2 br2 -> bl1 p1 p2 br2",
).flatten(1, 2)

Teff = einops.einsum(
    A_prod,
    get_local_term(g=0.6),
    A_prod.conj(),
    "bl phys br, phys physc, blc physc brc -> bl blc br brc",
)
Teff = einops.rearrange(Teff, "bl blc br brc -> (bl blc) (br brc)")
print(t.trace(Teff))

# %% Testing the order of contraction is correct

pauli_z = t.tensor([[1, 0], [0, -1]], dtype=t.complex128)
pauli_x = t.tensor([[0, 1], [1, 0]], dtype=t.complex128)

# do the exact computation without flattening indices
energy_zz = einops.einsum(
    A,
    A,
    pauli_z,
    pauli_z,
    A.conj(),
    A.conj(),
    "b1 p1 b2, b2 p2 b3, p1 p1c, p2 p2c, b1c p1c b2c, b2c p2c b3c -> b1 b1c b3 b3c",
)
energy_zz = einops.rearrange(energy_zz, "b1 b1c b3 b3c -> (b1 b1c) (b3 b3c)")

energy_xe = einops.einsum(
    A,
    A,
    pauli_x,
    t.eye(d_phys, dtype=t.complex128),
    A.conj(),
    A.conj(),
    "b1 p1 b2, b2 p2 b3, p1 p1c, p2 p2c, b1c p1c b2c, b2c p2c b3c -> b1 b1c b3 b3c",
)
energy_xe = einops.rearrange(energy_xe, "b1 b1c b3 b3c -> (b1 b1c) (b3 b3c)")

energy_ex = einops.einsum(
    t.eye(d_phys, dtype=t.complex128),
    A,
    A,
    pauli_x,
    A.conj(),
    A.conj(),
    "p1 p1c, b1 p1 b2, b2 p2 b3, p2 p2c, b1c p1c b2c, b2c p2c b3c -> b1 b1c b3 b3c",
)
energy_ex = einops.rearrange(energy_ex, "b1 b1c b3 b3c -> (b1 b1c) (b3 b3c)")

energy = -(energy_zz + 1 / 2 * 0.6 * energy_xe + 1 / 2 * 0.6 * energy_ex)
print(t.trace(energy))

# These should be the same
assert t.allclose(
    t.trace(energy), t.trace(Teff)
), "The trace of the energy is not the same as the trace of the effective term"

# %% Computing energy and left/right fixed point vectors

transfer_matrix = einops.einsum(
    A,
    A.conj(),
    "bl p br, blc p brc -> bl blc br brc",
)
transfer_matrix = einops.rearrange(
    transfer_matrix, "bl blc br brc -> (bl blc) (br brc)"
)
# Do an eigendecomposition
U, S, V = t.linalg.svd(transfer_matrix, full_matrices=False)
transfer_matrix /= S[0]
S_normalized = S / S[0]
left_fixed_point = U.H[0]
right_fixed_point = V.H[:, 0]

# print(transfer_matrix - U @ t.diag(S_normalized.to(t.complex128)) @ V)
# first singular value
# should be one right?

print(left_fixed_point.unsqueeze(0) @ transfer_matrix @ right_fixed_point.unsqueeze(1))


# %% Compute the iMPS ground state using fixed point calculation


def compute_energy(A_real, A_imag, g, n_site=None):
    A = A_real + 1j * A_imag
    A_prod = einops.einsum(
        A,
        A,
        "bl1 p1 b, b p2 br2 -> bl1 p1 p2 br2",
    ).flatten(1, 2)
    transfer_matrix = einops.einsum(
        A,
        A.conj(),
        "bl p br, blc p brc -> bl blc br brc",
    )
    transfer_matrix = einops.rearrange(
        transfer_matrix, "bl blc br brc -> (bl blc) (br brc)"
    )
    # Do an eigendecomposition
    U, S, Vh = t.linalg.svd(transfer_matrix, full_matrices=False)
    assert t.allclose(U @ t.diag(S.to(t.complex128)) @ Vh, transfer_matrix)
    # transfer_matrix /= S[0]
    # S_normalized = S / S[0]

    A_prod /= S[0]  # normalize the matrix product state

    Teff = einops.einsum(
        A_prod,
        get_local_term(g=g),
        A_prod.conj(),
        "bl phys br, phys physc, blc physc brc -> bl blc br brc",
    )
    # can normalize the matrix product state by normalizing the largest singular value of the transfer matrix to 1.
    Teff = einops.rearrange(Teff, "bl blc br brc -> (bl blc) (br brc)")

    if n_site is None:
        left_fixed_point = U.H[0]
        right_fixed_point = Vh.H[:, 0]
        return t.real(
            left_fixed_point.unsqueeze(0) @ Teff @ right_fixed_point.unsqueeze(1)
        ).squeeze()
    else:
        # we want exact energy here and we keep all the singular values
        S_normalized = S / S[0]
        total_energy = 0
        for i in range(Teff.shape[0]):
            total_energy += (
                S_normalized[i] ** (n_site - 2)
                * t.real(U.H[i].unsqueeze(0) @ Teff @ Vh.H[:, i].unsqueeze(1)).squeeze()
            )
        
        # print(S_normalized)
        # print(A_prod)
        return total_energy

        # The most basic way is just to do matrix power
        transfer_matrix_power = t.linalg.matrix_power(
            transfer_matrix / S[0], n_site - 2
        )
        # print(S[0])
        total_energy = t.real(t.trace(transfer_matrix_power @ Teff)).squeeze()
        return total_energy

    
    
    
def compute_energy_v1(A_real, A_imag, g, n_site=None):
    A = A_real + 1j * A_imag
    pass

import time

start_time = time.time()
print(compute_energy(A_real, A_imag, g=0.6, n_site=100))
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")

start_time = time.time()
print(compute_energy(A_real, A_imag, g=0.6))
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")


# %%

A = A_real + 1j * A_imag
transfer_matrix = einops.einsum(
    A,
    A.conj(),
    "bl p br, blc p brc -> bl blc br brc",
)
transfer_matrix = einops.rearrange(
    transfer_matrix, "bl blc br brc -> (bl blc) (br brc)"
)

U, S, Vh = t.linalg.svd(transfer_matrix, full_matrices=True)

# Finite case
N = 8
state_norm = t.trace(t.linalg.matrix_power(transfer_matrix/S[0], N))
A_real = A_real / state_norm**(1/(2*N)) / t.sqrt(S[0])
A_imag = A_imag / state_norm**(1/(2*N)) / t.sqrt(S[0])
A = A_real + 1j * A_imag

# %%
transfer_matrix = einops.einsum(
    A,
    A.conj(),
    "bl p br, blc p brc -> bl blc br brc",
)
transfer_matrix = einops.rearrange(
    transfer_matrix, "bl blc br brc -> (bl blc) (br brc)"
)

assert t.isclose(t.trace(t.linalg.matrix_power(transfer_matrix, N)), t.tensor(1.0, dtype=t.complex128))

# %% Calculate the optimal parameters
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn

A_real = nn.Parameter(t.randn(d_bond, d_phys, d_bond, dtype=t.float64))
A_imag = nn.Parameter(t.randn(d_bond, d_phys, d_bond, dtype=t.float64))

optimizer = optim.Adam([A_real, A_imag], lr=0.001)
pbar = tqdm(range(1000))
energy_history = []
for i in pbar:
    optimizer.zero_grad()
    energy = compute_energy(A_real, A_imag, g=0.6)
    energy.backward()
    optimizer.step()
    energy_history.append(energy.item())

    if i % 20 == 0:
        pbar.set_description(f"Energy: {energy.item():.4f}")

pbar = tqdm(range(300))
for i in pbar:
    optimizer.zero_grad()
    energy = compute_energy(A_real, A_imag, g=0.6, n_site=10)
    energy.backward()
    optimizer.step()
    energy_history.append(energy.item())




# %% Plot the energy history

import plotly.express as px
# import pandas as pd

px.line(energy_history, title="Energy history")


# %% sanity check of the energy

# Using this as the hot start of the algorithm for computing the finite case
# The finite-size scaling of the energy

# Get the state and compute its norm
print(A_real + 1j * A_imag)


