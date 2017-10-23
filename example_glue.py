import gobasis
import grid
import iodata
import numpy as np
from functools import wraps
from meanfield import *


#
# Decorators
#

def onetime(varname):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if getattr(self, varname) is not None:
                print("Trying to set a one-time attribute {varname}. Ignored")
                return
            func(self, *args, **kwargs)

        return wrapper

    return decorator


def cache(varname):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if getattr(self, varname) is None:
                val = func(self, *args, **kwargs)
                setattr(self, varname, val)
            return getattr(self, varname)

        return wrapper

    return decorator


def delayed(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self.init_finished:
            print("Instance must finalize instantiation before calling compute functions.")
            raise AttributeError
        return func(self, *args, **kwargs)

    return wrapper


def finalize(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        func(self, *args, **kwargs)
        self.init_finished = True

    return wrapper


#
# Options classes
#

class Options:
    # __init__ is for options. finish_init is for inputs used in calculations.
    def finish_init(self):
        raise NotImplementedError


class DelayedInit:
    init_finished = False


class Molecule(Options):
    def __init__(self, coords, atomic_numbers, pseudo_numbers=None, charge=None, multiplicity=None):
        self._coords = coords
        self._atomic_nums = atomic_numbers if isinstance(atomic_numbers, np.ndarray) else np.array(
            atomic_numbers)
        self._pseudo = pseudo_numbers if isinstance(pseudo_numbers, np.ndarray) else np.array(
            pseudo_numbers)

        self._charge = charge or 0

        # Some defaults can be determined within a class.
        if charge > 0:
            default_multiplicity = 3
        else:
            default_multiplicity = 1

        self._multiplicity = multiplicity or default_multiplicity

    @classmethod
    def from_file(cls, filename):
        # TODO: load from file
        mol = iodata.from_file(filename)
        return cls(mol.coords, mol.atomic_numbers)

    @property
    def coords(self):
        return self._coords

    @property
    def atomic_nums(self):
        return self._atomic_nums

    @property
    def pseudo_nums(self):
        return self._pseudo

    @property
    def stretched_bond(self):
        return self._check_bond_length()

    @property
    def negative_charge(self):
        return self._charge > 0

    @property
    def charge(self):
        return self._charge

    @property
    def multiplicity(self):
        return self._multiplicity

    def _check_bond_length(self):
        return False  # TODO: Replace dummy

    def _check_multiplicity_charge(self):
        return True  # TODO: Replace dummy

    @property
    def nelec(self):
        return (10, 10)  # TODO: Replace dummy

    @property
    def nelec_alpha(self):
        return self.nelec[0]

    @property
    def nelec_beta(self):
        return self.nelec[1]


class Basis(Options, DelayedInit):
    def __init__(self, bset=None):
        """Specify basis set options.
        bset can be a string containing the basis set name, or a nested tuple like ((int, string),)
            containing the atom index and basis set name respectively.
        """
        self._bset = bset
        self._olp = None
        self._kin = None
        self._er = None
        self._na = None
        self._nn = None

    @finalize
    def finish_init(self, coords, atomic_nums, pseudo_numbers):
        self._coords = coords
        self._numbers = atomic_nums
        self._pseudo = pseudo_numbers
        self._gobasis = gobasis.get_gobasis(coords, atomic_nums, self._bset)

    @property
    def bset(self):
        return self._bset

    @bset.setter
    @onetime("_bset")
    def bset(self, bset):
        self._bset = bset

    # Cached Properties

    @property
    @cache("_kin")
    @delayed
    def kinetic(self):
        return self._gobasis.compute_kinetic()

    @property
    @cache("_er")
    @delayed
    def electron_repulsion(self):
        return self._gobasis.compute_electron_repulsion()

    @property
    @cache("_na")
    @delayed
    def nuclear_attraction(self):
        return self._gobasis.compute_nuclear_attraction(self._coords, self._pseudo)

    @property
    @cache("_one")
    @delayed
    def one(self):
        return self.kinetic + self.nuclear_attraction

    @property
    @cache("_nn")
    @delayed
    def nucnuc(self):
        return gobasis.compute_nucnuc(self._coords, self._pseudo)

    @property
    @cache("_olp")
    @delayed
    def overlap(self):
        return self._gobasis.compute_overlap()

    @property
    @delayed
    def gobasis(self):
        return self._gobasis


class Guess(Options):
    pass


class CoreHamGuess(Guess):
    @finalize
    def finish_init(self, olp, one, *orbs):
        self._olp = olp
        self._one = one
        self._orbs = orbs

    def make_guess(self):
        # writes guess into orbs. No return value
        guess_core_hamiltonian(self._olp, self._one, *self._orbs)


class Grid(Options, DelayedInit):
    def __init__(self, accuracy='coarse'):
        self._accuracy = accuracy


class BeckeGrid(Grid):
    @finalize
    def finish_init(self, coords, atomic_numbers, pseudo_numbers):
        self._grid = grid.BeckeMolGrid(coords, atomic_numbers, pseudo_numbers,
                                       agspec=self._accuracy)

    @property
    @delayed
    def grid(self):
        return self._grid


class Orbitals(Options, DelayedInit):
    def __init__(self, spin="U"):
        self._spin = spin

    @finalize
    def finish_init(self, nbasis):
        if self.isRestricted:
            self._orbs = [Orbitals(nbasis), ]
        elif self.isUnrestricted:
            self._orbs = [Orbitals(nbasis), Orbitals(nbasis)]
        else:  # generalized
            self._orbs = [Orbitals(2 * nbasis)]

    @property
    def isRestricted(self):
        return self._spin == "R"

    @property
    def isUnrestricted(self):
        return self._spin == "U"

    @property
    def isGeneralized(self):
        return self._spin == "G"

    @property
    def orbs(self):
        return self._orbs

    @property
    def dms(self):
        return self._dms


#
# Solver classes
#

class SCF():
    def __init__(self, *args, **kwargs):
        self._solver = None
        raise NotImplementedError

    def solve(self, *args, **kwargs):
        self._solver(*args, **kwargs)


class DIIS(SCF):
    def __init__(self, *args, **kwargs):
        self._solver = DIIS(*args, **kwargs)


#
# Compute classes
#

class Method:
    def __init__(self):
        self._ham = None
        self._basis = None
        self._molecule = None

    @finalize
    def finish_init(self, coords, basis, scf):
        self._basis = basis
        self._coords = coords
        self._scf = scf

    @property
    @cache("_energy")
    @delayed
    def energy(self):
        return self._calculate_energy()

    @property
    @cache("_ham")
    @delayed
    def hamiltonian(self):
        return self._ham

    def _calculate_energy(self):
        raise NotImplementedError

    @delayed
    def _calculate_energy(self):
        self._scf.solve(self._ham, self._basis.olp, self._occ_model, *self._orb.orbs)
        return self._ham.energy


class HF(Options, Method, DelayedInit):
    @finalize
    def finish_init(self, molecule, basis, scf, occ_model, orb, grid):
        super().finish_init(molecule.coords, basis, scf)

        self._occ_model = occ_model
        self._orb = orb

        if orb.isRestricted:
            external = {'nn': basis.nucnuc}
            terms = [
                RTwoIndexTerm(basis.kinetic, 'kin'),
                RDirectTerm(basis.electron_repulsion, 'hartree'),
                RExchangeTerm(basis.electron_repulsion, 'x_hf'),
                RTwoIndexTerm(basis.nuclear_attraction, 'ne'),
            ]
            self._ham = REffHam(terms, external)
        elif orb.isUnrestricted:
            external = {'nn': basis.nucnuc}
            terms = [
                UTwoIndexTerm(basis.kinetic, 'kin'),
                UDirectTerm(basis.electron_repulsion, 'hartree'),
                UExchangeTerm(basis.electron_repulsion, 'x_hf'),
                UTwoIndexTerm(basis.nuclear_attraction, 'ne'),
            ]
            self._ham = UEffHam(terms, external)
        else:
            raise NotImplementedError


class DFT(Options, Method, DelayedInit):
    def __init__(self, xc=None, x=None, c=None, frac=None):
        if xc and (x or c or frac):
            print("Cannot specify xc and also x or c functionals or exchange fraction")
            raise ValueError

        if c and not x:
            print("Cannot specify correlation without exchange functional.")
            raise ValueError

        self._xc = xc
        self._x = x
        self._c = c
        self._frac = frac
        self._funcs = [i for i in (xc, x, c) if i is not None]

        super().__init__()

    @finalize
    def finish_init(self, molecule, basis, scf, occ_model, orb, grid):
        super().finish_init(molecule.coords, basis, scf)

        self._occ_model = occ_model
        self._orb = orb

        external = {'nn': basis.nucnuc}
        if orb.isRestricted:
            terms = self._get_terms(basis, grid, "R", *self._funcs)
            self._ham = REffHam(terms, external)
        elif orb.isUnrestricted:
            terms = self._get_terms(basis, grid, "U", *self._funcs)
            self._ham = UEffHam(terms, external)
        else:
            raise NotImplementedError

            # TODO: add smart XC selector

    _term_dict = {("R", "two"): RTwoIndexTerm, ("U", "two"): UTwoIndexTerm,
                  ("R", "direct"): RDirectTerm, ("U", "direct"): UDirectTerm,
                  ("R", "exchange"): RExchangeTerm, ("U", "exchange"): UExchangeTerm,
                  ("R", "grid"): RGridGroup, ("U", "grid"): UGridGroup,
                  ("R", "lda"): RLibXCLDA, ("U", "lda"): ULibXCLDA,
                  ("R", "gga"): RLibXCGGA, ("U", "gga"): ULibXCGGA,
                  ("R", "mgga"): RLibXCMGGA, ("U", "mgga"): ULibXCMGGA,
                  ("R", "hyb_gga"): RLibXCHybridGGA, ("U", "hyb_gga"): ULibXCHybridGGA,
                  ("R", "hyb_mgga"): RLibXCHybridMGGA, ("U", "hyb_mgga"): ULibXCHybridMGGA,
                  }

    def _get_terms(self, basis, grid, spin="U", *libxc_strings):
        libxc_terms = [self._parse_functional(s, spin) for s in libxc_strings]
        terms = [
            self._term_dict[(spin, "two")](basis.kinetic, 'kin'),
            self._term_dict[(spin, "direct")](basis.electron_repulsion, 'hartree'),
            self._term_dict[(spin, "grid")](basis.gobasis, grid, libxc_terms),
            self._term_dict[(spin, "two")](basis.nuclear_attraction, 'ne'),
        ]
        # Add exchange terms in for hybrid functionals
        for s, t in zip(libxc_strings, libxc_terms):
            if "hyb" in s[:3]:
                terms.append(self._term_dict[(spin, "exchange")](basis.electron_repulsion, 'x_hf',
                                                                 t.get_exx_fraction()))
        return terms

    def _parse_functional(self, s, spin):
        level, functional = s.split("_", 1)
        if level.lower() == "hyb":
            level2, functional = functional.split("_", 1)
            level = "_".join((level, level2))

        return self._term_dict[(spin, level)](functional)


class SmartCompute:
    def __init__(self, molecule, basis=None, guess=None, method=None, grid=None, scf=None,
                 orbs=None):
        # Initialize defaults if not provided
        basis = basis or Basis()
        guess = guess or CoreHamGuess()
        method = method or HF()
        scf = scf or DIIS()
        orbs = orbs or Orbitals()
        occ_model = AufbauOccModel(*molecule.nelec)

        if grid and isinstance(method, HF):
            print("Grid and HF are incompatible options.")
            raise ValueError

        # Grid and DFT imply one another
        if grid or isinstance(method, DFT):
            grid = grid or BeckeGrid()
            method = method or DFT()

        # Make smart defaults
        if molecule.stretched_bond or molecule.negative_charge:
            basis.bset = "6-311+G(2p,2d)"
        else:
            basis.bset = "6-311G(2p,2d)"

        # Compose the different option classes together to finalize instantiation
        basis.finish_init(molecule.coords, molecule.atomic_nums)
        guess.finish_init(basis.overlap, basis.one, *orbs.orbs)
        method.finish_init(molecule.coords, basis, scf, occ_model, orbs, grid)

        # store attributes
        self.basis = basis
        self.guess = guess
        self.method = method
        self.scf = scf
        self.orbs = orbs
        self.occ_model = occ_model
        self.grid = grid

    def __getattr__(self, item):
        # FIXME: by default, gets attributes from the method class. Maybe this isn't a good idea...
        # TODO: this breaks autocompletion in pycharm...
        return getattr(self._method, item)


#
# Example of use
#

# minimum use-case
mol = Molecule.from_file("water.xyz")
c = SmartCompute(mol)
print(c.energy)

# specify basis
mol = Molecule([0, 0, 0], [1])
basis = Basis("cc-pvtz")
c = SmartCompute(mol, basis)
print(c.energy)

# dft example
mol = Molecule.from_file("water.xyz")
method = DFT("LDA_X")
c = SmartCompute(mol, method=method)
print(c.energy)
