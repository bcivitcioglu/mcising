import unittest
from mcising.isinglattice import IsingLattice

class TestIsingLattice(unittest.TestCase):

    def test_initialization(self):
        lattice = IsingLattice(lattice_size=10, J1=1.0, J2=0.5, h=0.0)
        self.assertEqual(lattice.lattice_size, 10)
        self.assertEqual(lattice.J1, 1.0)
        self.assertEqual(lattice.J2, 0.5)
        self.assertEqual(lattice.h, 0.0)
        self.assertEqual(lattice.lattice_state.shape, (10, 10))

    def test_flip_spin(self):
        lattice = IsingLattice(lattice_size=10, J1=1.0, J2=0.5, h=0.0)
        initial_spin = lattice.lattice_state[0, 0]
        lattice.flip_spin(0, 0)
        self.assertEqual(lattice.lattice_state[0, 0], -initial_spin)

if __name__ == '__main__':
    unittest.main()
