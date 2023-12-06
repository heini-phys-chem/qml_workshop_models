
import torch
import numpy as np

from qml_lightning.representations.FCHL import FCHLCuda
from qml_lightning.models.hadamard_features import HadamardFeaturesModel

from ase.calculators.general import Calculator
from ase.atoms import Atoms

convback   = 1/23.016# forces to gradients (kcal/mol/A eV/A)
convback_E = 1/23.016 # convert energies kcal/mol to eV

class ML_calculator(Calculator):
  name = 'ML_Calculator'
  implemented_properties = ['energy', 'forces']


  def __init__(self, atoms, model, nuclear_charges, atomIDs, molIDs, atom_counts, foutput):
    self.model = model
    self.charges = nuclear_charges
    self.atomIDs = atomIDs
    self.molIDs = molIDs
    self.atom_counts = atom_counts
    self.foutput = foutput

  def append_mol(self, atoms):
    fout = self.foutput
    f = open(fout, 'a')

    NAMES = {1:'H', 7:'N', 6:'C', 8:'O'}
    geoms = atoms.get_positions()
    labels = atoms.get_atomic_numbers()

    f.write("{}\n\n".format(self.nAtoms))
    for i in range(len(geoms)):
        f.write("{} {} {} {}\n".format(NAMES[labels[i]], geoms[i][0], geoms[i][1], geoms[i][2]))

    f.close()


  def get_potential_energy(self,atoms=None,force_consistent=False):
    #cell = [[17.25837925, 0, 0], [0, 14.9458635, 0], [0, 0, 38.70524185]]
    cell = [[17.7112, 0, 0], [ 0, 17.7112, 0], [ 0, 0, 25.32]]
    cell_tensor = torch.tensor(cell, dtype=torch.float32).to('cuda')
    cell_tensor = cell_tensor.unsqueeze(0)
    cell_inv = torch.inverse(cell_tensor)

    xyz = torch.tensor(atoms.get_positions(), device='cuda', dtype=torch.float32, requires_grad=True)
    prediction_energy = self.model.forward(xyz[None], self.charges[None], self.atomIDs, self.molIDs, self.atom_counts, cell=cell_tensor, inv_cell=cell_inv)

    energy = prediction_energy.cpu().detach().numpy() * convback_E

    return energy[0]

  def get_forces(self, atoms=None):
    #cell = [[17.25837925, 0, 0], [0, 14.9458635, 0], [0, 0, 38.70524185]]
    cell = [[17.7112, 0, 0], [ 0, 17.7112, 0], [ 0, 0, 25.32]]
    cell_tensor = torch.tensor(cell, dtype=torch.float32).to('cuda')
    cell_tensor = cell_tensor.unsqueeze(0)
    cell_inv = torch.inverse(cell_tensor)

    xyz = torch.tensor(atoms.get_positions(), device='cuda', dtype=torch.float32, requires_grad=True)
    prediction_energy = self.model.forward(xyz[None], self.charges[None], self.atomIDs, self.molIDs, self.atom_counts, cell=cell_tensor, inv_cell=cell_inv)

    forces_torch, = torch.autograd.grad(-prediction_energy.sum(), xyz)
#    print(forces_torch.cpu().detach().numpy())
    Fss = forces_torch.cpu().detach().numpy() * convback
#    print(Fss)

    return Fss
