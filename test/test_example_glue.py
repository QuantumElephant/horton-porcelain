import numpy as np
from example_glue import Molecule

def test_stretched_bond_long():
    c = np.ndarray([[0., 0., 0.], []])

    Molecule._check_bond_length()