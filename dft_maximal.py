# Most complicated case of using Horton 3 with DFT. See HF maximal first.
# If (some variation of) options aren't specified here, it can't be done!

from horton import from_file, dft_solver, xc

#####
# General options (in addition to HF maximal)
#####

molecule = from_file("water.xyz")  # load molecule spec
xc_set = xc("pbe", ...)  # TODO: Come up with reasonable arguments
grid = BeckeMolGrid("fine")  # Don't generate grid. Only instantiate class
water_solver = dft_solver(molecule, xc=xc, grid=grid)  # generates ints and solves SCF as soon as it is instantiated

