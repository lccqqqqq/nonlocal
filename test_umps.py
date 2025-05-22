# %% Testing the Ising ground states obtained from the gradient descent
import os
os.chdir("/workspace")
from mps_utils import compute_energy_density, construct_umps
from spin_model import h_ising_model
import torch as t
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import pickle
import math
import numpy as np
import einops

# %% The ground state from gradient descent calculation

# Define parameters
g = 0.5
N = 8
d_phys = 2
d_bond = 16
learning_rate = 0.001
num_optimization_steps = 3000
device = "cuda" if t.cuda.is_available else "cpu"

params = nn.Parameter(t.randn(d_phys, int(d_bond * (d_bond+1) / 2), dtype=t.float64, device=device))
optimizer = optim.AdamW([params], lr=learning_rate, betas=(0.9, 0.999), eps=1e-8)

# Begin the training loop
def normalize_params(params):
    return params / params.norm(p=2)
    
data = {"energy_density": [], "tensor": []}
pbar = tqdm(range(num_optimization_steps))
for i in pbar:
    norm_params = normalize_params(params)
    A = construct_umps(norm_params, d_bond=d_bond)

    energy_density = compute_energy_density(A, g=0.5, n_site=8)
    energy_density.backward()
    optimizer.step()
    data["energy_density"].append(energy_density.item())
    data["tensor"].append(A.detach().cpu().numpy())
    
    if i % 20 == 0:
        pbar.set_description(f"Energy density: {energy_density.item()}")
        print(A[0][0][0].item())
    
# save data to file as a dataframe
df = pd.DataFrame(data)



def format_coupling(g: float) -> str:
    """
    Format coupling constant for filename, converting decimal to scientific notation with 'em'
    Example: 0.5 -> 5em-1
    """
    if g == 0:
        return "0"
    exponent = 0
    while g < 1:
        g *= 10
        exponent -= 1
    while g >= 10:
        g /= 10
        exponent += 1
    return f"{int(g)}em{exponent}"

coupling_str = format_coupling(g)
df.to_pickle(f"training_nsite_{N}_dbond_{d_bond}_coupling" + format_coupling(g) + ".pkl")

# %% Comparing with exact diagonalization

# energy baseline
g = 0.5
N = 8
k_set = [-(N-1)/N * math.pi + 2*i/N * math.pi for i in range(N)]
eps = lambda k: math.sqrt(1 + g**2 - 2*g*math.cos(k))
E_gs = -sum([eps(k) for k in k_set]) / N

# wavefunction baseline via exact diagonalization
H_ising = h_ising_model(J=1.0, Gm=g, h=0, N=N, bc='PBC', flg=0, return_local=False)
eigval, eigvec = np.linalg.eigh(H_ising)

assert abs(E_gs - eigval[0]/N) < 1e-10

ground_state = eigvec[:, 0]
assert abs(ground_state.T @ H_ising @ ground_state - eigval[0]) < 1e-10
assert abs(np.linalg.norm(ground_state) - 1) < 1e-10

# %% The metrics to evaluate are

# 1. the translation invariance has already been guaranteed by the uniformity of the MPs tensor
# 2. The overlap between the true ground state and the approximated round state
# 3. The difference between the energy and the target energy density of the true ground state

# Load the training data
df = pd.read_pickle(f"training_nsite_{N}_dbond_{d_bond}_coupling_{coupling_str}.pkl")

# Extract tensors from dataframe
tensors = df["tensor"].values
energy_densities = df["energy_density"].values

state_overlaps = []
for mps_tensor in tensors:
    tensor_state = mps_tensor
    for site in range(N-2):
        tensor_state = einops.einsum(
            tensor_state,
            mps_tensor,
            "... b, b phys bnew -> ... phys bnew"
        )
    
    # at last site
    tensor_state = einops.einsum(
        tensor_state,
        mps_tensor,
        "bl ... br, br phys bl -> ... phys",
    )
    mps_state = tensor_state.flatten()
    mps_state = mps_state / np.linalg.norm(mps_state)
    # calculate overlap
    state_overlaps.append(np.abs(np.dot(mps_state, ground_state)))


    
    
# %%
import plotly.express as px

px.line(energy_densities - E_gs, title="Energy density above ground state energy as training", log_x=False, log_y=True)

# %%

px.line(state_overlaps, title="Similarity to the ground state", log_x=False, log_y=True)

# Todo: what about the parity? Should we be enforcing *even* parity all the way?
