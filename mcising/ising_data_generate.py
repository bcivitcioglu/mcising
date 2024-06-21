# mcising/ising_data_generate.py

import numpy as np
import argparse
import montecarlo

def main(args):
    temperature = np.arange(args.T_init, args.T_final - args.T_step, -args.T_step)

    montecarlo.collect_monte_carlo_data(
        seed=args.seed,
        lattice_size=args.lattice_size,
        J1=args.j1, 
        J2=args.j2,
        h=0.0,
        num_scans=args.sweep_steps * (args.num_configs - 1),
        temperature=temperature,
        thermalization_scans=args.thermalization_scans,
        frequency_sweeps_to_collect_magnetization=args.sweep_steps,
        calculate_correlation=args.calculate_correlation
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Ising model data using Monte Carlo simulation.")
    parser.add_argument("seed", type=int, help="Random seed for the simulation")
    parser.add_argument("lattice_size", type=int, help="Size of the lattice")
    parser.add_argument("num_configs", type=int, help="Number of configurations to generate")
    parser.add_argument("j1", type=float, help="Interaction parameter J1")
    parser.add_argument("j2", type=float, help="Interaction parameter J2")
    parser.add_argument("--T_init", type=float, default=4.0, help="Initial temperature (default: 4.0)")
    parser.add_argument("--T_final", type=float, default=0.075, help="Final temperature (default: 0.075)")
    parser.add_argument("--T_step", type=float, default=0.025, help="Temperature step (default: 0.025)")
    parser.add_argument("--sweep_steps", type=int, default=1, help="Number of sweep steps (default: 1)")
    parser.add_argument("--thermalization_scans", type=int, default=3, help="Number of thermalization scans (default: 3)")
    parser.add_argument("--calculate_correlation", action="store_true", help="Calculate correlation function and length")

    args = parser.parse_args()

    if args.T_step <= 0:
        print("Error: T_step must be greater than 0")
    elif args.T_init <= args.T_final:
        print("Error: T_init must be greater than T_final")
    else:
        main(args)
