# Analyzing the training loss curves
# %%
import pandas as pd
import numpy as np
import plotly.express as px
import pickle
import re

data_path = "workspace/training_nsite_8_dbond_16_coupling_5em-1.pkl"
with open(data_path, "rb") as f:
    data = pickle.load(f)

print(data)

# %% Plotting the energy density

def parse_data_name(data_path):
    # parse the hyperparameters from the file name
    data_name = data_path.split("/")[-1].split(".")[0].split("_")
    # print(data_name)
    n_site = int(data_name[2])
    d_bond = int(data_name[4])
    coupling = re.sub(r'm', '', data_name[6])
    coupling = float(coupling)
    return n_site, d_bond, coupling

# %% plot the energy density and overlap with the ground state
n_site, d_bond, coupling = parse_data_name(data_path)

energy_density = data["energy_density"]


def get_ground_state_overlap(data):
    tensors = data["tensor"]
    ground_state = data["ground_state"]
    overlap = []
    for tensor in tensors:
        overlap.append(np.abs(np.dot(tensor.flatten(), ground_state)))
    return overlap


# %%



# %%
