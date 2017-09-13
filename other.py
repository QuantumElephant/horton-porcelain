# Other types of calculations

import numpy as np
from horton import from_file, basis_by_name, Kinetic, get_int

#####
# Save AO integrals without solving SCF
#####

molecule = from_file("water.xyz")  # load molecule spec
basis = basis_by_name(bset="cc-pvtz")
int_type = Kinetic()  # Just instantiate class. Don't calculate int
np.save("kin", get_int(molecule, basis, int_type))  # Calculate int
