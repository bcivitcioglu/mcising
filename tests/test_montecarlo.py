import unittest
from mcising.isinglattice import IsingLattice
from mcising.montecarlo import scan_lattice, monte_carlo_simulation

class TestMonteCarlo(unittest.TestCase):

    def setUp(self):
        self.lattice = IsingLattice(lattice_size=10, J1=1.0, J2=0.5, h=0.0)
        self.temperature = 1.0
        self.num_scans = 100
        self.frequency_sweeps_to_collect_magnetization = 10

    def test_scan_lattice(self):
        initial_energy = self.lattice.energy()
        scan_lattice(self.lattice, self.temperature)
        self.assertNotEqual(self.lattice.energy(), initial_energy)

    def test_monte_carlo_simulation_without_correlation(self):
        configs, energies, magnetizations, corrs, corr_lengths, distances = monte_carlo_simulation(
            self.lattice,
            self.temperature,
            self.num_scans,
            self.frequency_sweeps_to_collect_magnetization,
            calculate_correlation=False
        )
        self.assertIsNotNone(configs)
        self.assertIsNotNone(energies)
        self.assertIsNotNone(magnetizations)
        self.assertIsNone(corrs)
        self.assertIsNone(corr_lengths)
        self.assertIsNone(distances)

    def test_monte_carlo_simulation_with_correlation(self):
        configs, energies, magnetizations, corrs, corr_lengths, distances = monte_carlo_simulation(
            self.lattice,
            self.temperature,
            self.num_scans,
            self.frequency_sweeps_to_collect_magnetization,
            calculate_correlation=True
        )
        self.assertIsNotNone(configs)
        self.assertIsNotNone(energies)
        self.assertIsNotNone(magnetizations)
        self.assertIsNotNone(corrs)
        self.assertIsNotNone(corr_lengths)
        self.assertIsNotNone(distances)

if __name__ == '__main__':
    unittest.main()
