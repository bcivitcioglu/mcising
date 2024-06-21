import numpy as np
import matplotlib.pyplot as plt
import random

class IsingLattice:
    """
    A class to represent an Ising model lattice.
    
    Attributes
    ----------
    lattice_size : int
        The size of the lattice.
    num_sites : int
        The total number of sites in the lattice.
    J1 : float
        Interaction parameter for nearest neighbors.
    J2 : float
        Interaction parameter for next nearest neighbors.
    h : float
        External magnetic field.
    lattice_state : np.ndarray
        The state of the lattice.
    """
    
    def __init__(self, lattice_size, J1, J2, h):
        """
        Initializes the lattice with random spins.
        
        Parameters
        ----------
        lattice_size : int
            The size of the lattice.
        J1 : float
            Interaction parameter for nearest neighbors.
        J2 : float
            Interaction parameter for next nearest neighbors.
        h : float
            External magnetic field.
        """
        self.lattice_size = lattice_size
        self.num_sites = lattice_size * lattice_size
        self.J1 = J1
        self.J2 = J2
        self.h = h
        
        # Randomly initialize the lattice with -1 and 1
        lattice_state = np.random.choice([-1, 1], size=(self.lattice_size, self.lattice_size))
        self.lattice_state = lattice_state

    def plot_lattice(self, print_info=False): 
        """
        Plot the current lattice configuration.
        
        Parameters
        ----------
        print_info : bool, optional
            If True, prints the lattice information. Default is False.
        """
        plt.figure()
        plt.imshow(self.lattice_state, cmap='gray')
        plt.title("Ising Lattice")
        plt.show()
        if print_info:
            self.print_info()

    def print_info(self):
        """
        Print information about the lattice.
        """
        print(f"Lattice size: {self.lattice_size} x {self.lattice_size}. J1: {self.J1}, J2: {self.J2}, h: {self.h}")

    def flip_spin(self, i, j):
        """
        Flip the spin at the given site (i, j).
        
        Parameters
        ----------
        i : int
            Row index of the site.
        j : int
            Column index of the site.
        """
        self.lattice_state[i, j] *= -1

    def spin_energy(self, i, j):
        """
        Calculate the energy of the spin at the given site (i, j).
        
        Parameters
        ----------
        i : int
            Row index of the site.
        j : int
            Column index of the site.
        
        Returns
        -------
        float
            The energy of the spin at the given site.
        """
        spin_ij = self.lattice_state[i, j]

        # Periodic boundary conditions
        sum_neighbouring_spins = (
            self.lattice_state[(i + 1) % self.lattice_size, j] +
            self.lattice_state[i, (j + 1) % self.lattice_size] +
            self.lattice_state[(i - 1) % self.lattice_size, j] +
            self.lattice_state[i, (j - 1) % self.lattice_size]
        )

        sum_second_neighbouring_spins = (
            self.lattice_state[(i + 1) % self.lattice_size, (j + 1) % self.lattice_size] +
            self.lattice_state[(i + 1) % self.lattice_size, (j - 1) % self.lattice_size] +
            self.lattice_state[(i - 1) % self.lattice_size, (j + 1) % self.lattice_size] +
            self.lattice_state[(i - 1) % self.lattice_size, (j - 1) % self.lattice_size]
        )

        interaction_term = (
            -self.J1 * spin_ij * sum_neighbouring_spins +
            -self.J2 * spin_ij * sum_second_neighbouring_spins
        )
        
        if self.h != 0:
            magnetic_field_term = -self.h * spin_ij
            return interaction_term + magnetic_field_term
        
        return interaction_term

    def energy(self):
        """
        Calculate the total energy of the lattice.
        
        Returns
        -------
        float
            The total energy of the lattice.
        """
        E = 0.0
        for i in np.arange(self.lattice_size):
            for j in np.arange(self.lattice_size):
                E += self.spin_energy(i, j)
        E /= 2.0 * self.num_sites
        if self.h != 0:
            E -= self.h * np.sum(self.lattice_state) / self.num_sites
        return E

    def magnetization(self):
        """
        Calculate the net magnetization of the lattice.
        
        Returns
        -------
        float
            The net magnetization of the lattice.
        """
        return np.sum(self.lattice_state) / self.num_sites

    def correlation_function(self, plot=False):
        """
        Calculate the correlation function of the lattice.
        
        Parameters
        ----------
        plot : bool, optional
            If True, plots the correlation function. Default is False.
        
        Returns
        -------
        tuple
            The correlation function and the distances.
        """
        counter = 0
        correlation_function = np.zeros(self.num_sites**2)
        r_sq = np.zeros(self.num_sites**2).astype(int)

        for i in np.arange(self.lattice_size):
            for j in np.arange(self.lattice_size):
                for k in np.arange(i, self.lattice_size):
                    check_var = 0
                    if i == k:
                        check_var = j
                    for l in np.arange(check_var, self.lattice_size):
                        x_distance = abs(j - l)
                        y_distance = abs(i - k)
                        
                        if x_distance > self.lattice_size / 2:
                            x_distance = abs(self.lattice_size - x_distance)
                        if y_distance > self.lattice_size / 2:
                            y_distance = abs(self.lattice_size - y_distance)
                            
                        distance = x_distance**2 + y_distance**2
                        
                        r_sq[counter] = distance
                        correlation_function[counter] = self.lattice_state[i, j] * self.lattice_state[k, l]
                        
                        counter += 1
        
        corr = correlation_function[:counter]
        dist = r_sq[:counter]

        sort_ind = np.argsort(dist)
        sorted_d = np.sort(dist)
        sorted_c = corr[sort_ind]
        
        unique_d, unique_indices_d = np.unique(sorted_d, return_index=True)
        averaged_c = np.zeros(unique_d.size)

        for i in np.arange(averaged_c.size - 1):
            denom = unique_indices_d[i + 1] - unique_indices_d[i]
            averaged_c[i] = np.sum(sorted_c[unique_indices_d[i]:unique_indices_d[i + 1]]) / denom

        denom = unique_indices_d[-1] - unique_indices_d[-2]
        averaged_c[-1] = np.sum(sorted_c[unique_indices_d[-2]:unique_indices_d[-1]]) / denom
        
        if plot:
            plt.plot(np.sqrt(unique_d), averaged_c)
            plt.xlabel("Distance")
            plt.ylabel("Correlation Function")
            plt.title("Correlation Function vs Distance")
            plt.show()

        return averaged_c - self.magnetization()**2, np.sqrt(unique_d)
