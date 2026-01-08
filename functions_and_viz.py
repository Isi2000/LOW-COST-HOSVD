"""
LOW-COST-HOSVD: Functions and Visualization Tools
Contains all decomposition methods and visualization utilities for tensor analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorly as tl
import time
from tqdm import tqdm
from matplotlib.animation import FuncAnimation, PillowWriter


# ============================================================================
# DECOMPOSITION FUNCTIONS
# ============================================================================

def select_rank_by_ratio(array, varepsilon):
    """
    Select rank by energy ratio criterion.

    Parameters:
    -----------
    array : ndarray
        Array of singular values or eigenvalues
    varepsilon : float
        Energy retention ratio (e.g., 0.99 for 99%)

    Returns:
    --------
    int : Selected rank
    """
    sorted_array = np.sort(array)[::-1]
    total_energy = np.sum(sorted_array)
    cumulative_energy = np.cumsum(sorted_array)
    num_components = np.sum(cumulative_energy <= varepsilon * total_energy) + 1
    return min(num_components, len(array))


def sequentially_truncated_hosvd(tensor, varepsilon=0.99, time_it=False):
    """
    Sequentially Truncated HOSVD using Gram matrix method.

    Parameters:
    -----------
    tensor : ndarray
        Input tensor
    varepsilon : float
        Energy retention ratio (default: 0.99)
    time_it : bool
        Whether to return computation time

    Returns:
    --------
    core : ndarray
        Core tensor
    factors : list
        List of factor matrices
    time : float (optional)
        Computation time if time_it=True
    """
    start_t = time.time()
    factors = []
    current_tensor = tensor.copy()

    for mode in range(current_tensor.ndim):
        unfolded = tl.unfold(current_tensor, mode)
        g_matrix = unfolded @ unfolded.T
        eigenvalues, eigenvectors = np.linalg.eigh(g_matrix)
        eigenvalues = np.sqrt(np.abs(eigenvalues))
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        trunc_idx = select_rank_by_ratio(eigenvalues, varepsilon)
        U_trunc = eigenvectors[:, :trunc_idx].real
        current_tensor = tl.tenalg.mode_dot(current_tensor, U_trunc.T, mode)
        factors.append(U_trunc)

    core = current_tensor
    elapsed_time = time.time() - start_t

    if time_it:
        return core, factors, elapsed_time
    else:
        return core, factors


def subsampled_hosvd_gram(tensor, subsample_factor=2, varepsilon=0.99, time_it=False):
    """
    Subsampled HOSVD using Gram matrix method.

    Parameters:
    -----------
    tensor : ndarray
        Input tensor
    subsample_factor : int
        Subsampling factor (default: 2)
    varepsilon : float
        Energy retention ratio (default: 0.99)
    time_it : bool
        Whether to return computation time

    Returns:
    --------
    core : ndarray
        Core tensor
    factors : list
        List of factor matrices
    time : float (optional)
        Computation time if time_it=True
    """
    start_t = time.time()
    factors = []

    for mode in range(tensor.ndim):
        # Subsample all modes except current one
        slicing = tuple(
            slice(None) if d == mode else slice(None, None, subsample_factor)
            for d in range(tensor.ndim)
        )
        subsampled = tensor[slicing]
        unfolded = tl.unfold(subsampled, mode)

        g_matrix = unfolded @ unfolded.T
        eigenvalues, eigenvectors = np.linalg.eigh(g_matrix)
        eigenvalues = np.sqrt(np.abs(eigenvalues))
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        trunc_idx = select_rank_by_ratio(eigenvalues, varepsilon)
        U_trunc = eigenvectors[:, :trunc_idx].real
        factors.append(U_trunc)

    core = tl.tenalg.multi_mode_dot(
        tensor,
        [f.T for f in factors],
        modes=list(range(tensor.ndim))
    )

    elapsed_time = time.time() - start_t

    if time_it:
        return core, factors, elapsed_time
    else:
        return core, factors


def subsampled_hosvd_svd(tensor, subsample_factor=2, varepsilon=0.99, time_it=False):
    """
    Subsampled HOSVD using direct SVD method.

    Parameters:
    -----------
    tensor : ndarray
        Input tensor
    subsample_factor : int
        Subsampling factor (default: 2)
    varepsilon : float
        Energy retention ratio (default: 0.99)
    time_it : bool
        Whether to return computation time

    Returns:
    --------
    core : ndarray
        Core tensor
    factors : list
        List of factor matrices
    time : float (optional)
        Computation time if time_it=True
    """
    start_t = time.time()
    factors = []

    for mode in range(tensor.ndim):
        # Subsample all modes except current one
        slicing = tuple(
            slice(None) if d == mode else slice(None, None, subsample_factor)
            for d in range(tensor.ndim)
        )
        subsampled = tensor[slicing]
        unfolded = tl.unfold(subsampled, mode)

        U, S, Vh = np.linalg.svd(unfolded, full_matrices=False)
        trunc_idx = select_rank_by_ratio(S, varepsilon)
        U_trunc = U[:, :trunc_idx].real
        factors.append(U_trunc)

    core = tl.tenalg.multi_mode_dot(
        tensor,
        [f.T for f in factors],
        modes=list(range(tensor.ndim))
    )

    elapsed_time = time.time() - start_t

    if time_it:
        return core, factors, elapsed_time
    else:
        return core, factors


def subsampled_hosvd(test_tensor, sampling_ratio, time_it=False, sv_threshold=1e-03):
    """
    Original subsampled HOSVD implementation from tensor_decompositions.py
    """
    t_start = time.time()
    factors_sub = []

    for mode in range(test_tensor.ndim):
        unfolded = tl.unfold(test_tensor, mode)
        m, n = unfolded.shape
        n_samples = int(n * sampling_ratio)
        sample_indices = np.random.choice(n, size=n_samples, replace=False)
        sampled_matrix = unfolded[:, sample_indices]

        U, S, _ = np.linalg.svd(sampled_matrix, full_matrices=False)

        threshold = sv_threshold * S[0]
        rank = np.sum(S >= threshold)
        rank = max(1, rank)
        U_truncated = U[:, :rank]
        factors_sub.append(U_truncated)

    core_sub = tl.tenalg.multi_mode_dot(
        test_tensor,
        [U.T for U in factors_sub],
        modes=[mode for mode in range(test_tensor.ndim)]
    )

    t_subsampled = time.time() - t_start

    if time_it:
        return core_sub, factors_sub, t_subsampled
    else:
        return core_sub, factors_sub


def low_cost_hosvd(test_tensor, time_it=False, sampling_fraction=0.5,
                   mode_fraction=0.95, special_mode=0):
    """
    Low-cost HOSVD implementation from tensor_decompositions.py
    """
    t_start = time.time()
    sampling_mode_size = test_tensor.shape[special_mode]
    n_samples = max(1, int(sampling_mode_size * sampling_fraction))
    sampled_indices = np.random.choice(sampling_mode_size, size=n_samples, replace=False)
    slicing = [slice(None)] * test_tensor.ndim
    slicing[special_mode] = sampled_indices
    subsampled_tensor_lc = test_tensor[tuple(slicing)]

    factors_lc = []

    for mode in range(test_tensor.ndim):
        unfolded = tl.unfold(subsampled_tensor_lc, mode)

        if mode != special_mode:
            U, S, _ = np.linalg.svd(unfolded, full_matrices=False)
            total_modes = min(U.shape)
            rank = max(1, int(total_modes * mode_fraction))
            U_truncated = U[:, :rank]
        else:
            U_red, S, V_red = np.linalg.svd(unfolded, full_matrices=False)
            Q, R = np.linalg.qr(U_red)
            U_red = U_red @ np.linalg.inv(R)
            Q, R = np.linalg.qr(V_red.T)
            V_red = (V_red.T @ np.linalg.inv(R)).T
            ss = U_red.T @ unfolded @ V_red.T
            ss_sign = np.sign(np.diag(ss))
            V_red = V_red.T @ np.diag(ss_sign)
            V_red = V_red.T
            non_sampled_unfolded = tl.unfold(test_tensor, mode)
            U = non_sampled_unfolded @ V_red.T @ np.diag(1/S)
            Q, _ = np.linalg.qr(U)
            U = Q
            total_modes = min(U.shape)
            rank = max(1, int(total_modes * mode_fraction))
            U_truncated = U[:, :rank]
        factors_lc.append(U_truncated)

    core_lc = tl.tenalg.multi_mode_dot(
        test_tensor,
        [U.T for U in factors_lc],
        modes=[mode for mode in range(test_tensor.ndim)]
    )

    t_lc = time.time() - t_start

    if time_it:
        return core_lc, factors_lc, t_lc
    else:
        return core_lc, factors_lc


def reconstruct_tensor(core, factors):
    """Reconstruct tensor from core and factor matrices."""
    return tl.tenalg.multi_mode_dot(core, factors, modes=[i for i in range(len(factors))])


# ============================================================================
# ERROR COMPUTATION FUNCTIONS
# ============================================================================

def compute_error(test_tensor, reconstruction):
    """Compute relative Frobenius norm error."""
    return np.linalg.norm(np.subtract(test_tensor, reconstruction)) / np.linalg.norm(test_tensor)


def compute_error_normalized_by_species(test_tensor, reconstruction, species_axis=2):
    """
    Compute reconstruction error normalized by mean value per species.

    Parameters:
    -----------
    test_tensor : ndarray
        Original tensor
    reconstruction : ndarray
        Reconstructed tensor
    species_axis : int
        Axis corresponding to species dimension (default: 2)

    Returns:
    --------
    overall_error : float
        Overall normalized error
    per_species_errors : ndarray
        Array of normalized errors per species
    """
    error = np.abs(test_tensor - reconstruction)
    n_species = test_tensor.shape[species_axis]
    per_species_errors = np.zeros(n_species)

    for species_idx in range(n_species):
        slicing = [slice(None)] * test_tensor.ndim
        slicing[species_axis] = species_idx

        species_data = test_tensor[tuple(slicing)]
        species_error = error[tuple(slicing)]

        species_mean = np.mean(np.abs(species_data))

        if species_mean > 0:
            species_rmse = np.sqrt(np.mean(species_error**2))
            per_species_errors[species_idx] = species_rmse / species_mean
        else:
            per_species_errors[species_idx] = 0.0

    overall_error = np.mean(per_species_errors)

    return overall_error, per_species_errors


def compute_compression_factor(original_tensor, core, factors):
    """Compute compression factor of decomposition."""
    original_size = original_tensor.size
    decomp_size = core.size + sum(U.size for U in factors)
    return original_size / decomp_size


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_species_comparison_bar(per_species_errors_orig, per_species_errors_log,
                                 species_names, overall_error_orig, overall_error_log,
                                 save_path=None):
    """
    Create bar plot comparing normalized errors by species.

    Parameters:
    -----------
    per_species_errors_orig : ndarray
        Per-species errors for original tensor
    per_species_errors_log : ndarray
        Per-species errors for log tensor
    species_names : list
        List of species names
    overall_error_orig : float
        Overall error for original tensor
    overall_error_log : float
        Overall error for log tensor
    save_path : str, optional
        Path to save the figure
    """
    n_species = len(species_names)
    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(n_species)
    width = 0.35

    bars1 = ax.bar(x - width/2, per_species_errors_orig, width,
                   label='Original Tensor', color='#2E86AB', alpha=0.8)
    bars2 = ax.bar(x + width/2, per_species_errors_log, width,
                   label='Log Tensor', color='#A23B72', alpha=0.8)

    ax.set_xlabel('Species', fontsize=12, fontweight='bold')
    ax.set_ylabel('Normalized Error (RMSE / Mean)', fontsize=12, fontweight='bold')
    ax.set_title('Normalized Error by Species: Original vs Log Tensor stHOSVD',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(species_names, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_error_histograms(tensor_orig, reconstruction_orig, reconstruction_log,
                          species_names, save_path=None):
    """
    Plot error histograms per species in 2x4 grid.

    Parameters:
    -----------
    tensor_orig : ndarray
        Original tensor
    reconstruction_orig : ndarray
        Reconstruction from original tensor
    reconstruction_log : ndarray
        Reconstruction from log tensor
    species_names : list
        List of species names
    save_path : str, optional
        Path to save the figure
    """
    n_species = len(species_names)
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    axes = axes.ravel()

    error_orig = (tensor_orig - reconstruction_orig)
    error_log = (tensor_orig - reconstruction_log)

    for i in range(n_species):
        ax = axes[i]
        err_o = error_orig[:, :, i, :].ravel()
        err_l = error_log[:, :, i, :].ravel()

        vmin = min(err_o.min(), err_l.min())
        vmax = max(err_o.max(), err_l.max())
        maxabs = max(abs(vmin), abs(vmax)) if (vmin != vmax) else max(abs(vmin), 1e-8)
        bins = np.linspace(-maxabs, maxabs, 100)

        ax.hist(err_o, bins=bins, color='#2E86AB', alpha=0.7, label='Original')
        ax.hist(err_l, bins=bins, color='#A23B72', alpha=0.6, label='Log')

        rmse_o = np.sqrt(np.mean(err_o**2))
        rmse_l = np.sqrt(np.mean(err_l**2))

        ax.set_title(f"{species_names[i]} — RMSE Orig: {rmse_o:.3e}, Log: {rmse_l:.3e}",
                     fontsize=11)
        ax.set_yscale('log')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_xlim(-maxabs, maxabs)

        if i == 0:
            ax.legend(fontsize=9)

    for ax in axes[n_species:]:
        ax.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_spatial_comparison(tensor_orig, reconstruction_orig, reconstruction_log,
                            species_names, time_idx=0, save_path=None):
    """
    Create 3x8 subplot visualization for all species at a given time step.

    Parameters:
    -----------
    tensor_orig : ndarray
        Original tensor
    reconstruction_orig : ndarray
        Reconstruction from original tensor
    reconstruction_log : ndarray
        Reconstruction from log tensor
    species_names : list
        List of species names
    time_idx : int
        Time index to visualize
    save_path : str, optional
        Path to save the figure
    """
    n_species = len(species_names)

    # Compute global color scales for each species
    vmax_per_species = []
    for species_idx in range(n_species):
        delta_orig_all = tensor_orig[:, :, species_idx, :] - reconstruction_orig[:, :, species_idx, :]
        delta_log_all = tensor_orig[:, :, species_idx, :] - reconstruction_log[:, :, species_idx, :]
        vmax = max(np.max(np.abs(delta_orig_all)), np.max(np.abs(delta_log_all)))
        vmax_per_species.append(vmax)

    error_st_orig = compute_error(tensor_orig, reconstruction_orig)
    error_st_log = compute_error(tensor_orig, reconstruction_log)

    fig, axes = plt.subplots(3, n_species, figsize=(24, 9))

    fig.suptitle(f'stHOSVD Comparison: Original vs Log Transform (t={time_idx})\n' +
                 f'Original Tensor Error: {error_st_orig:.6f} | Log Tensor Error: {error_st_log:.6f}',
                 fontsize=16, fontweight='bold', y=0.98)

    for species_idx in range(n_species):
        original = tensor_orig[:, :, species_idx, time_idx]
        reconstructed_orig = reconstruction_orig[:, :, species_idx, time_idx]
        reconstructed_log = reconstruction_log[:, :, species_idx, time_idx]

        delta_orig = original - reconstructed_orig
        delta_log = original - reconstructed_log

        vmax = vmax_per_species[species_idx]

        # Row 1: Original data
        ax1 = axes[0, species_idx]
        im1 = ax1.imshow(original, cmap='viridis', aspect='auto', origin='lower')
        ax1.set_title(f'{species_names[species_idx]}', fontsize=11, fontweight='bold')
        if species_idx == 0:
            ax1.set_ylabel('Original Data', fontsize=11, fontweight='bold')
        ax1.set_xticks([])
        ax1.set_yticks([])
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        # Row 2: Delta (original)
        ax2 = axes[1, species_idx]
        im2 = ax2.imshow(delta_orig, cmap='RdBu_r', aspect='auto', origin='lower',
                         vmin=-vmax, vmax=vmax)
        if species_idx == 0:
            ax2.set_ylabel('Delta (Original)', fontsize=11, fontweight='bold')
        ax2.set_xticks([])
        ax2.set_yticks([])
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        # Row 3: Delta (log)
        ax3 = axes[2, species_idx]
        im3 = ax3.imshow(delta_log, cmap='RdBu_r', aspect='auto', origin='lower',
                         vmin=-vmax, vmax=vmax)
        if species_idx == 0:
            ax3.set_ylabel('Delta (Log)', fontsize=11, fontweight='bold')
        ax3.set_xticks([])
        ax3.set_yticks([])
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def create_comparison_animation(tensor_orig, reconstruction_orig, reconstruction_log,
                                species_names, output_filename='comparison_animation.gif',
                                fps=10):
    """
    Create animated GIF showing comparison across all time steps.

    Parameters:
    -----------
    tensor_orig : ndarray
        Original tensor
    reconstruction_orig : ndarray
        Reconstruction from original tensor
    reconstruction_log : ndarray
        Reconstruction from log tensor
    species_names : list
        List of species names
    output_filename : str
        Output filename for the GIF
    fps : int
        Frames per second for the animation
    """
    n_species = len(species_names)
    n_snapshots = tensor_orig.shape[3]

    print(f"Creating animated GIF with {n_snapshots} frames...")

    # Compute global color scales
    vmax_per_species = []
    for species_idx in range(n_species):
        delta_orig_all = tensor_orig[:, :, species_idx, :] - reconstruction_orig[:, :, species_idx, :]
        delta_log_all = tensor_orig[:, :, species_idx, :] - reconstruction_log[:, :, species_idx, :]
        vmax = max(np.max(np.abs(delta_orig_all)), np.max(np.abs(delta_log_all)))
        vmax_per_species.append(vmax)

    error_st_orig = compute_error(tensor_orig, reconstruction_orig)
    error_st_log = compute_error(tensor_orig, reconstruction_log)

    # Create figure and axes
    fig, axes = plt.subplots(3, n_species, figsize=(24, 9))

    # Initialize image objects
    images = []
    for species_idx in range(n_species):
        row_images = []
        for row_idx in range(3):
            ax = axes[row_idx, species_idx]
            if row_idx == 0:
                im = ax.imshow(np.zeros((tensor_orig.shape[0], tensor_orig.shape[1])),
                              cmap='viridis', aspect='auto', origin='lower')
            else:
                im = ax.imshow(np.zeros((tensor_orig.shape[0], tensor_orig.shape[1])),
                              cmap='RdBu_r', aspect='auto', origin='lower',
                              vmin=-vmax_per_species[species_idx],
                              vmax=vmax_per_species[species_idx])
            row_images.append(im)

            if row_idx == 0:
                ax.set_title(f'{species_names[species_idx]}', fontsize=11, fontweight='bold')
            if species_idx == 0:
                if row_idx == 0:
                    ax.set_ylabel('Original Data', fontsize=11, fontweight='bold')
                elif row_idx == 1:
                    ax.set_ylabel('Delta (Original)', fontsize=11, fontweight='bold')
                else:
                    ax.set_ylabel('Delta (Log)', fontsize=11, fontweight='bold')

            ax.set_xticks([])
            ax.set_yticks([])
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        images.append(row_images)

    title = fig.suptitle('', fontsize=16, fontweight='bold', y=0.98)

    def update(frame):
        title.set_text(
            f'stHOSVD Comparison: Original vs Log Transform (t={frame}/{n_snapshots-1})\n' +
            f'Original Tensor Error: {error_st_orig:.6f} | Log Tensor Error: {error_st_log:.6f}'
        )

        for species_idx in range(n_species):
            original = tensor_orig[:, :, species_idx, frame]
            reconstructed_orig = reconstruction_orig[:, :, species_idx, frame]
            reconstructed_log = reconstruction_log[:, :, species_idx, frame]

            delta_orig = original - reconstructed_orig
            delta_log = original - reconstructed_log

            images[species_idx][0].set_array(original)
            images[species_idx][0].set_clim(vmin=original.min(), vmax=original.max())
            images[species_idx][1].set_array(delta_orig)
            images[species_idx][2].set_array(delta_log)

        return [img for row in images for img in row] + [title]

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    anim = FuncAnimation(fig, update, frames=n_snapshots, interval=100, blit=False)

    print(f"Saving animation to {output_filename}...")
    writer = PillowWriter(fps=fps)
    anim.save(output_filename, writer=writer)
    print(f"Animation saved successfully!")

    plt.close()

    print(f"Animation created with {n_snapshots} frames at {fps} FPS")
    print(f"Duration: {n_snapshots/fps:.1f} seconds")


def print_species_comparison_table(per_species_errors_orig, per_species_errors_log,
                                   species_names, overall_error_orig, overall_error_log):
    """
    Print formatted comparison table of normalized errors by species.

    Parameters:
    -----------
    per_species_errors_orig : ndarray
        Per-species errors for original tensor
    per_species_errors_log : ndarray
        Per-species errors for log tensor
    species_names : list
        List of species names
    overall_error_orig : float
        Overall error for original tensor
    overall_error_log : float
        Overall error for log tensor
    """
    print("\n" + "="*80)
    print("NORMALIZED ERROR BY SPECIES (RMSE / Mean per species)")
    print("="*80)

    print(f"\n{'Species':<10} {'Original Tensor':<20} {'Log Tensor':<20} {'Improvement':<15}")
    print("-"*80)

    for species_idx in range(len(species_names)):
        improvement = (per_species_errors_orig[species_idx] - per_species_errors_log[species_idx]) / \
                      per_species_errors_orig[species_idx] * 100
        print(f"{species_names[species_idx]:<10} {per_species_errors_orig[species_idx]:<20.6f} "
              f"{per_species_errors_log[species_idx]:<20.6f} {improvement:>+13.2f}%")

    print("-"*80)
    print(f"{'OVERALL':<10} {overall_error_orig:<20.6f} {overall_error_log:<20.6f} "
          f"{(overall_error_orig - overall_error_log)/overall_error_orig * 100:>+13.2f}%")
    print("="*80)
