# minimal examples of DFT. See HF minimal first.

import numpy as np
from horton import from_file, from_inchi, from_pubchem, from_coordinates, dft_solver

####
# From (any) file type
####

molecule = from_file("water.xyz")  # load molecule spec
water_solver = dft_solver(molecule)  # generates ints and runs HF as soon as it is instantiated
print("energy: ", water_solver.energy)  # print stored energy
np.save("ones", water_solver.ones_mo)  # save stored MOs