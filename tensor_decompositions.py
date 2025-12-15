import numpy as np
import matplotlib.pyplot as plt
import tensorly as tl
import time
from tqdm import tqdm


def subsampled_hosvd(test_tensor, sampling_ratio, time_it=False):
    t_start = time.time()
    factors_sub = []
    
    for mode in range(test_tensor.ndim):
        unfolded = tl.unfold(test_tensor, mode)
        m, n = unfolded.shape
        n_samples = int(n * sampling_ratio)
        sample_indices = np.random.choice(n, size=n_samples, replace=False)
        sampled_matrix = unfolded[:, sample_indices]
        U, _, _ = np.linalg.svd(sampled_matrix, full_matrices=False)
        factors_sub.append(U)
    
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


def sequentially_truncated_hosvd(test_tensor, energy_threshold=0.99, time_it=False):
    #DA RIVEDERE!!!!!!!!!!!!!!!!!
    t_start = time.time()
    factors_st = []
    current_tensor = test_tensor.copy()
    
    for mode in range(test_tensor.ndim):
        unfolded = tl.unfold(current_tensor, mode)
        U, S, Vt = np.linalg.svd(unfolded, full_matrices=False)
        cumulative_energy = np.cumsum(S**2) / np.sum(S**2)
        rank = np.searchsorted(cumulative_energy, energy_threshold) + 1
        rank = min(rank, len(S))
        U_truncated = U[:, :rank]
        factors_st.append(U_truncated)
        current_tensor = tl.tenalg.mode_dot(current_tensor, U_truncated.T, mode)
    
    core_st = current_tensor
    t_st = time.time() - t_start
    
    if time_it:
        return core_st, factors_st, t_st
    else:
        return core_st, factors_st


def low_cost_hosvd(test_tensor, time_it=False):
    t_start = time.time()
    subsampled_tensor_lc = test_tensor[:, :, ::2]
    factors_lc = []
    
    for mode in range(test_tensor.ndim):
        unfolded = tl.unfold(subsampled_tensor_lc, mode)
        
        if mode != test_tensor.ndim - 1:
            U, _, _ = np.linalg.svd(unfolded, full_matrices=False)
        else:
            U_red, sigma, V_red = np.linalg.svd(unfolded, full_matrices=False)
            Q, R = np.linalg.qr(U_red)
            U_red = U_red @ np.linalg.inv(R)
            Q, R = np.linalg.qr(V_red.T)
            V_red = (V_red.T @ np.linalg.inv(R)).T
            ss = U_red.T @ unfolded @ V_red.T
            ss_sign = np.sign(np.diag(ss))
            V_red = V_red.T @ np.diag(ss_sign)
            V_red = V_red.T
            non_sampled_unfolded = tl.unfold(test_tensor, mode)
            U = non_sampled_unfolded @ V_red.T @ np.diag(1/sigma)
            Q, _ = np.linalg.qr(U)
            U = Q
        
        factors_lc.append(U)
    
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
    return tl.tenalg.multi_mode_dot(core, factors, modes=[i for i in range(len(factors))])


def compute_error(test_tensor, reconstruction):
    return np.linalg.norm(np.subtract(test_tensor, reconstruction)) / np.linalg.norm(test_tensor)


def compute_compression_factor(original_tensor, core, factors):
    original_size = original_tensor.size
    decomp_size = core.size + sum(U.size for U in factors)
    return original_size / decomp_size