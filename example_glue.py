import gobasis
import grid
import iodata
import numpy as np
import periodic
from abc import ABCMeta
from meanfield import *

from decorators import finalize, cache, onetime, delayed


#
# Options classes
#

class Options(metaclass=ABCMeta):
    """
    Users instantiate child classes of this class to specify which options they want
    for calculations.
    """

    # __init__ is for options. finish_init is for inputs used in calculations.
    def finish_init(self):
        raise NotImplementedError


class Molecule(Options):
    def __init__(self, coords, atomic_numbers, pseudo_numbers=None, charge=None, multiplicity=None):
        self._coords = coords if isinstance(coords, np.ndarray) else np.array(coords)
        self._atomic_nums = atomic_numbers if isinstance(atomic_numbers, np.ndarray) else np.array(
            atomic_numbers, dtype=int)
        self._pseudo = pseudo_numbers if isinstance(pseudo_numbers, np.ndarray) else np.array(
            pseudo_numbers, dtype=int)

        if not self._check_multiplicity_charge(multiplicity, charge):
            print("Non-sensical combination of multiplicity and charge given.")
            raise ValueError

        self._charge = charge or 0

        # Some defaults can be determined within a class.
        if charge > 0:
            default_multiplicity = 3
        else:
            default_multiplicity = 1

        self._multiplicity = multiplicity or default_multiplicity

    def finish_init(self):
        pass

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
    def stretched_bonds(self):
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

    @staticmethod
    def _check_bond_length(self, z: np.ndarray, r: np.ndarray):  # TODO: write unit test.
        def cov_distance(self, za: int, zb: int):
            return (periodic[za].cov_radius + periodic[zb].cov_radius)

        def bond_length(self, ra: np.ndarray, rb: np.ndarray):
            return np.linalg.norm(ra - rb)

        stretched_bonds = set()

        for i, za, ra in enumerate(zip(z, r)):
            for j, zb, rb in enumerate(zip(z[i:], r[i:])):
                if 1.2 * cov_distance(za, zb) < bond_length(ra, rb) < 1.3 * cov_distance(za, zb):
                    stretched_bonds.add()
        return False

    def _check_multiplicity_charge(self, multiplicity, charge):
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


class Basis(Options):
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

    @property
    @delayed
    def nbasis(self):
        return self._gobasis.nbasis

    @property
    @delayed
    def gobasis(self):
        return self._gobasis

    # Cached Properties

    @property
    @cache("_kin")
    @delayed
    def kinetic(self):
        x = self._gobasis.compute_kinetic()
        x.flags.writable = False
        return x

    @property
    @cache("_er")
    @delayed
    def electron_repulsion(self):
        x = self._gobasis.compute_electron_repulsion()
        x.flags.writable = False
        return x

    @property
    @cache("_na")
    @delayed
    def nuclear_attraction(self):
        x = self._gobasis.compute_nuclear_attraction(self._coords, self._pseudo)
        x.flags.writable = False
        return x

    @property
    @cache("_one")
    @delayed
    def one(self):
        x = self.kinetic + self.nuclear_attraction
        x.flags.writable = False
        return x

    @property
    @cache("_nn")
    @delayed
    def nucnuc(self):
        return gobasis.compute_nucnuc(self._coords, self._pseudo)

    @property
    @cache("_olp")
    @delayed
    def overlap(self):
        x = self._gobasis.compute_overlap()
        x.flags.writable = False
        return x


class Guess(Options):
    pass


class CoreHamGuess(Guess):
    @finalize
    def finish_init(self, olp, one, *orbs):
        self._olp = olp
        self._one = one
        self._orbs = orbs

    @delayed
    def make_guess(self):
        # writes guess into orbs. No return value
        guess_core_hamiltonian(self._olp, self._one, *self._orbs)


class Grid(Options):
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


class Orbitals(Options):
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

class Method(Options):
    def __init__(self):
        self._ham = None
        self._basis = None
        self._molecule = None
        self._occ_model = None
        self._orb = None

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


class HF(Method):
    @finalize
    def finish_init(self, coords, basis, scf, occ_model, orb, grid):
        super().finish_init(coords, basis, scf)

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


class DFT(Method):
    def __init__(self, functionals: str = None, frac: float = None):  # TODO: 1 frac per functional?
        """

        Parameters
        ----------
        functionals
            A comma separated string of libxc functionals. Use the libxc full functional name.
            Some additional functionals are recognized for convenience's sake. (ie BP86)
        frac
            A float for the exchange fraction used in a functional.
        """
        self._funcs = [i.upper().strip() for i in functionals.split(",")]
        # TODO: transform common aliases into libxc functionals

        if frac and "HYB_" not in functionals:
            print("Cannot specify exchange fraction without hybrid type functional")
            raise ValueError

        if "_K_" in functionals:
            print("Kinetic functionals are not supported yet.")
            raise NotImplementedError

        self._frac = frac

        super().__init__()

    @property
    def xc(self):
        return [i for i in self._funcs if "_XC_" in i or i.endswith("_XC")]

    @property
    def x(self):
        return [i for i in self._funcs if "_X_" in i or i.endswith("_X")]

    @property
    def c(self):
        return [i for i in self._funcs if "_C_" in i or i.endswith("_C")]

    @property
    def functionals(self):
        return self._funcs

    @functionals.setter
    @onetime("_funcs")
    def functionals(self, s):
        self._funcs = [i.upper().strip() for i in s.split(",")]

    @finalize
    def finish_init(self, coords, basis, scf, occ_model, orb, grid):
        super().finish_init(coords, basis, scf)

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
        #
        # Initialize defaults if not provided
        #
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

        #
        # Make smart defaults
        #
        if molecule.stretched_bonds or molecule.negative_charge:
            basis.bset = "6-311+G(2p,2d)"
        else:
            basis.bset = "6-311G(2p,2d)"

        # Select xc for DFT:
        if isinstance(method, DFT):
            if molecule.stretched_bonds:
                method.functionals = "HYB_GGA_XC_B3LYP"
            elif max(molecule.atomic_nums) <= 10 \
                    and 3 not in molecule.atomic_nums \
                    and 4 not in molecule.atomic_nums:
                method.functionals = "HYB_MGGA_XC_M06"
            elif max(molecule.atomic_nums) > 36:
                print("Cannot compute Z > 36. We do not have relativisitic corrections yet.")
                raise NotImplementedError
            else:
                method.functionals = "GGA_X_B88, GGA_C_LYP"

        #
        # Compose the different option classes together to finalize instantiation
        #
        basis.finish_init(molecule.coords, molecule.atomic_nums)
        guess.finish_init(basis.overlap, basis.one, *orbs.orbs)
        if grid:
            grid.finish_init(molecule.coords, molecule.atomic_nums, molecule.pseudo_nums)
        orbs.finish_init(basis.nbasis)
        method.finish_init(molecule.coords, basis, scf, occ_model, orbs, grid)

        #
        # store attributes
        #
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

# restricted calculation
mol = Molecule.from_file("water.xyz")
orb = Orbitals("R")
c = SmartCompute(mol, orbs=orb)
print(c.energy)

# dft example
mol = Molecule.from_file("water.xyz")
method = DFT("LDA_X")
c = SmartCompute(mol, method=method)
print(c.energy)
