# Most complicated case of using Horton 3 with HF.
# If (some variation of) options aren't specified here, it can't be done!

from horton import from_file, from_pubchem, from_coordinates, hf_solver, basis_by_name, DIISSolver, \
    MinAO, basis_by_contraction, basis_by_atom, CDIISSolver

#####
# General options
#####

molecule = from_file("water.xyz", units="angstrom")  # load molecule spec
basis = basis_by_name(bset="cc-pvtz", dfset="weighend")  # Doesn't generate integrals yet (no molecule spec)
scf = DIISSolver(tolerance=0.05, max_iter=50)
initial_guess = MinAO()
water_solver = hf_solver(molecule, basis, scf, initial_guess)  # generates ints and solves SCF as soon as it is instantiated

#####
# Basis options
#####

molecule = from_coordinates(coords=[[0, 0, 0], ], numbers=[1])
basis = basis_by_contraction(...) # TODO: Come up with sensible arguments
water_solver = hf_solver(molecule, basis)  # generates ints and runs HF as soon as it is instantiated

molecule = from_coordinates(coords=[[0, 0, 0], [0,1,0]], numbers=[1,1])
basis = basis_by_atom(([0], "cc-pvdz"), ([1], "aug-ccpvtz"))
water_solver = hf_solver(molecule, basis=basis)  # generates ints and runs HF as soon as it is instantiated

#####
# Solver options
#####
molecule = from_pubchem("2519")  # load molecule spec
scf = CDIISSolver(tolerance=1e-5, max_iter=50)  # Or other variants of solver
water_solver = hf_solver(molecule, scf=scf)
