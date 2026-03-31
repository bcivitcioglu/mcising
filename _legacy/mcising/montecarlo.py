import numpy as np
import matplotlib.pyplot as plt
import random
import pickle as pkl
import imageio
import time
import os
from .isinglattice import IsingLattice

INF_TEMP = 100

def get_index(data):
    """
    Get the index where data falls below a threshold.
    
    Parameters
    ----------
    data : np.ndarray
        The array of data.
        
    Returns
    -------
    int
        The index where data falls below 10^-8.
    """
    for i in range(data.size):
        if data[i] < 10**(-8):
            return i
    return data.size

def data2cut(data, index):
    """
    Cut data up to a given index.
    
    Parameters
    ----------
    data : np.ndarray
        The array of data.
    index : int
        The index to cut data.
        
    Returns
    -------
    np.ndarray
        The cut data.
    """
    return data[:index]

def distance2cut(distance, index):
    """
    Cut distances up to a given index.
    
    Parameters
    ----------
    distance : np.ndarray
        The array of distances.
    index : int
        The index to cut distances.
        
    Returns
    -------
    np.ndarray
        The cut distances.
    """
    return distance[:index]

def get_cor_len(cor_func, dist, index):
    """
    Calculate the correlation length.
    
    Parameters
    ----------
    cor_func : np.ndarray
        The correlation function values.
    dist : np.ndarray
        The distances.
    index : int
        The index up to which the data is considered.
        
    Returns
    -------
    float
        The correlation length.
    """
    cor_func_cut = data2cut(cor_func, index)
    distance_cut = distance2cut(dist, index)

    correlation_negative_check = np.sum(cor_func_cut * (distance_cut ** 2))
    correlation_zero_check = np.sum(6 * cor_func_cut)
    
    if correlation_zero_check == 0:
        return 0
    elif correlation_negative_check / correlation_zero_check < 0:
        return 0
    else:
        return np.sqrt(correlation_negative_check / correlation_zero_check)

def scan_lattice(ising_lattice, temperature):
    """
    Perform a single Metropolis scan of the lattice.
    
    Parameters
    ----------
    ising_lattice : IsingLattice
        The Ising lattice.
    temperature : float
        The temperature of the system.
    """
    for _ in range(ising_lattice.num_sites):
        i = random.randint(0, ising_lattice.lattice_size - 1)
        j = random.randint(0, ising_lattice.lattice_size - 1)

        energy_initial = ising_lattice.spin_energy(i, j)
        ising_lattice.flip_spin(i, j)
        energy_final = ising_lattice.spin_energy(i, j)
        energy_change = energy_final - energy_initial
        ising_lattice.flip_spin(i, j)
        
        if temperature != 0:
            if energy_change <= 0 or random.uniform(0, 1) <= np.exp(-energy_change / temperature):
                ising_lattice.flip_spin(i, j)

def calculate_corr_size(lattice_size):
    """
    Calculate CORR_SIZE dynamically based on lattice size.
    
    Parameters
    ----------
    lattice_size : int
        The size of the lattice.
        
    Returns
    -------
    int
        The calculated CORR_SIZE.
    """
    unique_distances = set()

    for i in range(lattice_size):
        for j in range(lattice_size):
            for k in range(lattice_size):
                for l in range(lattice_size):
                    x_distance = min(abs(i - k), lattice_size - abs(i - k))
                    y_distance = min(abs(j - l), lattice_size - abs(j - l))
                    distance_squared = x_distance**2 + y_distance**2
                    unique_distances.add(distance_squared)
    
    return len(unique_distances)

def monte_carlo_simulation(ising_lattice, temperature, num_scans, frequency_sweeps_to_collect_magnetization, plot_result=False, print_info=False, calculate_correlation=False):
    """
    Perform Monte Carlo simulation on the Ising lattice.
    
    Parameters
    ----------
    ising_lattice : IsingLattice
        The Ising lattice.
    temperature : float
        The temperature of the system.
    num_scans : int
        The number of scans to perform.
    frequency_sweeps_to_collect_magnetization : int
        The frequency of sweeps to collect magnetization.
    plot_result : bool, optional
        Whether to plot the result. Default is False.
    print_info : bool, optional
        Whether to print information. Default is False.
    calculate_correlation : bool, optional
        Whether to calculate correlation function and length. Default is False.
        
    Returns
    -------
    tuple
        Lattice configurations, energy records, magnetization records, correlation function records, correlation length records, distances.
    """
    start_time = time.time()
    
    if print_info:
        ising_lattice.print_info()
    
    total_num_records = int(num_scans / frequency_sweeps_to_collect_magnetization) + 1
    energy_records = np.zeros(total_num_records)
    magnetization_records = np.zeros(total_num_records)
    
    corr_size = calculate_corr_size(ising_lattice.lattice_size) if calculate_correlation else 0
    correlation_function_records = np.zeros((total_num_records, corr_size)) if calculate_correlation else None
    correlation_length_records = np.zeros(total_num_records) if calculate_correlation else None
    
    increment_records = 0
    
    lattice_configs = np.zeros((total_num_records, ising_lattice.lattice_size, ising_lattice.lattice_size))

    for k in range(num_scans + frequency_sweeps_to_collect_magnetization):
        scan_lattice(ising_lattice, temperature)
        if k % frequency_sweeps_to_collect_magnetization == 0:
            energy_records[increment_records] = ising_lattice.energy()
            magnetization_records[increment_records] = ising_lattice.magnetization()
            lattice_configs[increment_records] = ising_lattice.lattice_state
            
            if calculate_correlation:
                correlations, distances = ising_lattice.correlation_function(False)
                correlation_function_records[increment_records] = correlations[:corr_size]
                
                index = get_index(correlations)
                correlation_length_records[increment_records] = get_cor_len(correlations, distances, index)
            
            increment_records += 1
            print(f"{increment_records} / {total_num_records} samples saved.")
    
    print(f"For temperature= {temperature}, MC simulation executed in: {round(time.time() - start_time, 2)} seconds")
    
    if plot_result:
        ising_lattice.plot_lattice()
    
    return (
        lattice_configs, 
        energy_records, 
        magnetization_records, 
        correlation_function_records if calculate_correlation else None, 
        correlation_length_records if calculate_correlation else None, 
        distances if calculate_correlation else None
    )

def dir_name(lattice_size, J1, J2, h, temperature):
    return f'SQ_L_{lattice_size}_J1_{J1:.3f}_J2_{J2:.3f}_h_{h:.3f}_T_{temperature:.3f}'

def file_name(lattice_size, J1, J2, h, temperature, seed):
    return f'SQ_L_{lattice_size}_J1_{J1:.3f}_J2_{J2:.3f}_h_{h:.3f}_T_{temperature:.3f}_s_{seed}'

def write_to_sub_directory(quantity, dir_name, file_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    with open(os.path.join(dir_name, file_name), "wb") as file:
        pkl.dump(quantity, file)

def write_txt_files(quantity, dir_name, file_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    np.savetxt(os.path.join(dir_name, file_name), quantity, fmt='%1.3f')

def save_image_to_sub_directory(data, directory_name, file_name):
    if not os.path.exists(directory_name):
        os.mkdir(directory_name)
    imageio.imwrite(os.path.join(directory_name, f"{file_name}.png"), data)

def thermalize(ising_lattice, num_scans, from_T, to_T):
    """
    Thermalize the lattice from a given temperature to a target temperature.
    
    Parameters
    ----------
    ising_lattice : IsingLattice
        The Ising lattice.
    num_scans : int
        The number of scans for thermalization.
    from_T : float
        The initial temperature.
    to_T : float
        The target temperature.
    """
    print(f"Equilibrating to T = {to_T:.2f} starting from T = {from_T:.2f}")
    for k in np.linspace(from_T, to_T, num=num_scans):
        scan_lattice(ising_lattice, k)
    print(f"Reached T={to_T:.2f}. Beginning to collect data.")

def collect_monte_carlo_data(seed, lattice_size, J1, J2, h, num_scans, temperature, thermalization_scans, frequency_sweeps_to_collect_magnetization, calculate_correlation=False):
    random.seed(seed)
    print(f"Lattice size: {lattice_size} x {lattice_size}, J1= {J1}, J2= {J2}, h= {h}, SEED= {seed}\n")
    temperature = np.append(INF_TEMP, temperature)
    
    if 0 in temperature:
        raise ValueError("Monte-Carlo does not work properly at T=0.")
    if np.any(temperature < 0):
        raise ValueError("Temperature cannot be a negative value.")
    
    ising_lattice = IsingLattice(lattice_size, J1, J2, h)
    num_temps = temperature.size
    
    for i in range(num_temps - 1):
        file_name_lattice = file_name(lattice_size, J1, J2, h, temperature[i+1], seed)
        dir_name_data = dir_name(lattice_size, J1, J2, h, temperature[i+1])
        
        total_num_configurations = int(num_scans / frequency_sweeps_to_collect_magnetization) + 1
        file_exists = [os.path.isfile(os.path.join(dir_name_data, f"{file_name_lattice}_n_{configs * frequency_sweeps_to_collect_magnetization}.pkl")) for configs in range(total_num_configurations)]
        
        if os.path.exists(dir_name_data) and not np.all(file_exists):
            print(f"{np.argwhere(np.array(file_exists) == False)[0][0]} Previous configurations for SEED = {seed} with L = {lattice_size} T = {temperature[i+1]:.2f} J1 = {J1} J2 = {J2} h = {h}\n")

        if np.all(file_exists):
            print(f"ALL requested configurations for SEED = {seed} with L = {lattice_size} T = {temperature[i+1]:.2f} J1 = {J1} J2 = {J2} h = {h} already exist!\n")
            continue
        
        thermalize(ising_lattice, thermalization_scans, temperature[i], temperature[i+1])
        print(f"START - MC simulation {i+1} / {num_temps-1}, T = {temperature[i+1]:.2f}")

        lattice_configs, energy_records, magnetization_records, correlation_function_records, correlation_length_records, distances = monte_carlo_simulation(
            ising_lattice, temperature[i+1], num_scans, frequency_sweeps_to_collect_magnetization, calculate_correlation=calculate_correlation)

        for img in range(total_num_configurations):
            file_name_img = f"{file_name_lattice}_n_{img * frequency_sweeps_to_collect_magnetization}"
            data_sample = {
                'configuration': lattice_configs[img],
                'energy': energy_records[img],
                'magnetization': magnetization_records[img],
                'correlation_length': correlation_length_records[img] if calculate_correlation else None,
                'correlation_function': correlation_function_records[img] if calculate_correlation else None,
                'distances': distances if calculate_correlation else None
            }

            txt_data = np.array([data_sample['energy'], data_sample['magnetization'], data_sample['correlation_length']])
            correlation_function_txt_data = np.array(data_sample['correlation_function']) if calculate_correlation else None
            write_to_sub_directory(data_sample, dir_name_data, f"{file_name_img}.pkl")
            save_image_to_sub_directory(lattice_configs[img].astype(np.uint8), dir_name_data, file_name_img)
            write_txt_files(txt_data, dir_name_data, f"{file_name_img}.txt")
            if calculate_correlation:
                write_txt_files(correlation_function_txt_data, dir_name_data, f"{file_name_img}_correlation_function.txt")
                write_txt_files(distances, dir_name_data, f"{file_name_img}_distances.txt")

        print(f"END --- MC simulation {i+1} / {num_temps-1}, T = {temperature[i+1]:.2f}\n")
