"""
LOW-COST-HOSVD: Main Execution Script
Loads combustion data and runs HOSVD decomposition analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import kagglehub
import json
from tqdm import tqdm
import os

from functions_and_viz import (
    sequentially_truncated_hosvd,
    subsampled_hosvd_gram,
    subsampled_hosvd_svd,
    compute_error,
    compute_error_normalized_by_species,
    compute_compression_factor,
    plot_species_comparison_bar,
    plot_error_histograms,
    plot_spatial_comparison,
    create_comparison_animation,
    print_species_comparison_table
)


def load_combustion_data(reynolds_numbers=[8000], subsample_x=5, subsample_y=5):
    """
    Load combustion data from Kaggle dataset.

    Parameters:
    -----------
    reynolds_numbers : list
        List of Reynolds numbers to load
    subsample_x : int
        Subsampling factor in x direction
    subsample_y : int
        Subsampling factor in y direction

    Returns:
    --------
    tensor_combustion : ndarray
        Loaded and subsampled tensor data
    metadata : dict
        Dataset metadata
    species_names : list
        List of species names
    """
    paths = [f"sharmapushan/hydrogen-jet-{re}" for re in reynolds_numbers]
    data_paths = [kagglehub.dataset_download(name) for name in paths]
    data_path = data_paths[0]

    with open(data_paths[0] + '/info.json') as f:
        metadata = json.load(f)

    Nx, Ny = metadata['global']['Nxyz']
    n_snapshots = metadata['global']['snapshots'] - 1
    Nx_sub = Nx // subsample_x
    Ny_sub = Ny // subsample_y

    component_names = ['YH', 'YH2', 'YO', 'YO2', 'YOH', 'YH2O', 'YHO2', 'YH2O2']
    species_names = [cname[1:] for cname in component_names]
    n_species = len(component_names)

    molar_masses = {
        'YH': 1.0, 'YH2': 2.0, 'YO': 8.0, 'YO2': 16.0,
        'YOH': 9.0, 'YH2O': 10.0, 'YHO2': 17.0, 'YH2O2': 18.0
    }

    file_key_map = {
        'YH': 'YH filename', 'YH2': 'YH2 filename',
        'YO': 'YO filename', 'YO2': 'YO2 filename',
        'YOH': 'YOH filename', 'YH2O': 'YH2O filename',
        'YHO2': 'YHO2 filename', 'YH2O2': 'YH2O2 filename'
    }

    print(f"Grid: {Nx}x{Ny}, Subsampled: {Nx_sub}x{Ny_sub}")

    tensor_ = np.zeros((Ny_sub, Nx_sub, n_species, n_snapshots))

    for t_idx in tqdm(range(n_snapshots), desc="Loading data"):
        for new_idx, (comp_name, orig_idx) in enumerate(zip(component_names, range(n_species))):
            filename_key = file_key_map[comp_name]
            filename = metadata['local'][t_idx][filename_key]
            data = np.fromfile(f"{data_path}/{filename}", dtype='<f4').reshape(Ny, Nx)
            molar_data = data / molar_masses[comp_name]
            tensor_[:, :, new_idx, t_idx] = molar_data[::subsample_x, ::subsample_y]

    print(f"Loaded tensor shape: {tensor_.shape}")
    print("Data loading complete!")

    return tensor_[:, :, :, :], metadata, species_names


def create_log_tensor(tensor, epsilon=1e-10):
    """
    Create log-transformed tensor.

    Parameters:
    -----------
    tensor : ndarray
        Original tensor
    epsilon : float
        Small constant to avoid log(0)

    Returns:
    --------
    tensor_log : ndarray
        Log-transformed tensor
    epsilon : float
        Epsilon value used (for inverse transform)
    """
    tensor_log = np.zeros_like(tensor)
    for i in range(tensor.shape[2]):
        tensor_log[:, :, i, :] = np.log(np.maximum(tensor[:, :, i, :], epsilon))
    print(f"Log tensor created with shape: {tensor_log.shape}")
    return tensor_log, epsilon


def run_comparison(tensor_combustion, tensor_combustion_log, epsilon,
                   species_names, varepsilon=0.999):
    """
    Run stHOSVD comparison on original vs log tensor.

    Parameters:
    -----------
    tensor_combustion : ndarray
        Original tensor
    tensor_combustion_log : ndarray
        Log-transformed tensor
    epsilon : float
        Epsilon used in log transform
    species_names : list
        List of species names
    varepsilon : float
        Energy retention ratio

    Returns:
    --------
    dict : Dictionary containing all results
    """
    print("\n" + "="*80)
    print("Running stHOSVD on ORIGINAL tensor")
    print("="*80)

    core_st_orig, factors_st_orig, time_orig = sequentially_truncated_hosvd(
        tensor_combustion, varepsilon=varepsilon, time_it=True
    )

    reconstruction_st_orig = np.tensordot(core_st_orig, factors_st_orig[0], axes=0)
    for i in range(1, len(factors_st_orig)):
        reconstruction_st_orig = np.tensordot(reconstruction_st_orig, factors_st_orig[i],
                                              axes=([i], [1]))
    reconstruction_st_orig = np.moveaxis(reconstruction_st_orig, 0, -1)

    # Correct reconstruction
    from functions_and_viz import reconstruct_tensor
    reconstruction_st_orig = reconstruct_tensor(core_st_orig, factors_st_orig)

    error_st_orig = compute_error(tensor_combustion, reconstruction_st_orig)
    compression_orig = compute_compression_factor(tensor_combustion, core_st_orig, factors_st_orig)

    print(f"Time: {time_orig:.2f} seconds")
    print(f"Core shape: {core_st_orig.shape}")
    print(f"Relative error: {error_st_orig:.6f}")
    print(f"Compression factor: {compression_orig:.2f}x")

    print("\n" + "="*80)
    print("Running stHOSVD on LOG tensor")
    print("="*80)

    core_st_log, factors_st_log, time_log = sequentially_truncated_hosvd(
        tensor_combustion_log, varepsilon=varepsilon, time_it=True
    )

    reconstruction_st_log_space = reconstruct_tensor(core_st_log, factors_st_log)
    reconstruction_st_log = np.exp(reconstruction_st_log_space)

    error_st_log = compute_error(tensor_combustion, reconstruction_st_log)
    compression_log = compute_compression_factor(tensor_combustion_log, core_st_log, factors_st_log)

    print(f"Time: {time_log:.2f} seconds")
    print(f"Core shape: {core_st_log.shape}")
    print(f"Relative error: {error_st_log:.6f}")
    print(f"Compression factor: {compression_log:.2f}x")

    # Compute normalized errors by species
    overall_error_orig, per_species_errors_orig = compute_error_normalized_by_species(
        tensor_combustion, reconstruction_st_orig, species_axis=2
    )
    overall_error_log, per_species_errors_log = compute_error_normalized_by_species(
        tensor_combustion, reconstruction_st_log, species_axis=2
    )

    # Print comparison table
    print_species_comparison_table(
        per_species_errors_orig, per_species_errors_log,
        species_names, overall_error_orig, overall_error_log
    )

    return {
        'core_orig': core_st_orig,
        'factors_orig': factors_st_orig,
        'reconstruction_orig': reconstruction_st_orig,
        'error_orig': error_st_orig,
        'time_orig': time_orig,
        'compression_orig': compression_orig,
        'core_log': core_st_log,
        'factors_log': factors_st_log,
        'reconstruction_log': reconstruction_st_log,
        'error_log': error_st_log,
        'time_log': time_log,
        'compression_log': compression_log,
        'per_species_errors_orig': per_species_errors_orig,
        'per_species_errors_log': per_species_errors_log,
        'overall_error_orig': overall_error_orig,
        'overall_error_log': overall_error_log
    }


def main():
    """Main execution function."""
    print("="*80)
    print("LOW-COST HOSVD: Combustion Data Analysis")
    print("="*80)

    # Load data
    tensor_combustion, metadata, species_names = load_combustion_data(
        reynolds_numbers=[8000],
        subsample_x=5,
        subsample_y=5
    )

    # Create log tensor
    tensor_combustion_log, epsilon = create_log_tensor(tensor_combustion)

    # Run comparison
    results = run_comparison(
        tensor_combustion,
        tensor_combustion_log,
        epsilon,
        species_names,
        varepsilon=0.999
    )

    # Create visualizations
    print("\n" + "="*80)
    print("Creating visualizations...")
    print("="*80)

    # Bar plot
    plot_species_comparison_bar(
        results['per_species_errors_orig'],
        results['per_species_errors_log'],
        species_names,
        results['overall_error_orig'],
        results['overall_error_log'],
        save_path='species_comparison_bar.png'
    )

    # Error histograms
    plot_error_histograms(
        tensor_combustion,
        results['reconstruction_orig'],
        results['reconstruction_log'],
        species_names,
        save_path='error_histograms.png'
    )

    # Spatial comparison
    plot_spatial_comparison(
        tensor_combustion,
        results['reconstruction_orig'],
        results['reconstruction_log'],
        species_names,
        time_idx=50,
        save_path='spatial_comparison.png'
    )

    # Create animation
    create_comparison_animation(
        tensor_combustion,
        results['reconstruction_orig'],
        results['reconstruction_log'],
        species_names,
        output_filename='stHOSVD_comparison_animation.gif',
        fps=10
    )

    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)


if __name__ == "__main__":
    main()
