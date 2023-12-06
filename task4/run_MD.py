#!/usr/bin/env python3


import numpy as np
import os

import torch
from qml_lightning.representations.FCHL import FCHLCuda
from qml_lightning.models.hadamard_features import HadamardFeaturesModel

from ase.md.langevin import Langevin
from ase.io import read, write

from ase.md import MDLogger
from ase import units

from ase.calculators import qmlightning
from ase.calculators import ff
from ase.calculators import gaussian
from ase.calculators.emt import EMT

import time

cuda = torch.cuda.is_available()
n_gpus = 1 if cuda else None
device = torch.device('cuda' if cuda else 'cpu')
seed = 1337
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


model  = torch.jit.load('/home/heinen/QMLworkshop/task4/model_sorf.pt')
initial_atoms = read('test.xyz')

coords = initial_atoms.get_positions()
nuclear_charges = initial_atoms.get_atomic_numbers()
nuclear_charges = np.repeat(nuclear_charges[np.newaxis,:], coords.shape[0], axis=0)
unique_z = np.unique(np.concatenate(nuclear_charges)).astype(int)

test_coordinates = [initial_atoms.get_positions()]
test_charges = [initial_atoms.get_atomic_numbers()]

xyz = torch.tensor(test_coordinates[0], device='cuda', dtype=torch.float32, requires_grad=True)
charges = torch.tensor(test_charges[0], device='cuda', dtype=torch.float32)
atomIDs =torch.arange(xyz.shape[0],device='cuda', dtype=torch.int32)
molIDs = torch.zeros(xyz.shape[0], device='cuda',dtype=torch.int32)
atom_counts= torch.tensor([xyz.shape[0]], device='cuda', dtype=torch.int32)


initial_atoms.calc = qmlightning.ML_calculator(initial_atoms, model, charges, atomIDs, molIDs, atom_counts, 'tmp.out')

# Set up the Langevin dynamics
temperature = 300.0  # Kelvin
friction_coefficient = 1e-3  # Langevin thermostat friction coefficient
time_step = 0.5  # femtoseconds
total_steps = 10000  # Total number of MD steps

cmd = "cp energies.out energies_tmp.out"
os.system(cmd)
cmd = "rm -f energies.out"
os.system(cmd)

start = time.time()

dyn = Langevin(initial_atoms, timestep=time_step * units.fs, temperature_K=temperature, friction=friction_coefficient)

# Run the MD simulation
for step in range(total_steps):
#for step in range(steps[i]):
    dyn.run(1)  # Run 1 MD step
    # Check if it's a step that you want to save (e.g., every 100 steps)
    if step % 100 == 0:
        # Create a unique filename based on the step number
        filename = f'xyz/coords_{step:05d}.xyz'

        # Write the coordinates to the XYZ file
        write(filename, initial_atoms, format='xyz')
        print("Temp: {}, Step: {}, E: {}".format(temperature, step, initial_atoms.get_potential_energy()))
        end = time.time()
        fout = open("energies.out", 'a')
        fout.write("{:.6f}\n".format(initial_atoms.get_potential_energy()))
        fout.close()

# Save the final structure to an XYZ file
write('final_structure.xyz', initial_atoms)
end = time.time()
print('Time MD run: {:.8f}'.format((end-start)/60.))
