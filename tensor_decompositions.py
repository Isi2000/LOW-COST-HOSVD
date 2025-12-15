import numpy as np
import matplotlib.pyplot as plt
import tensorly as tl
import time
from tqdm import tqdm

def subsampled_hosvd(test_tensor, sampling_ratio, time_it=False, sv_threshold=1e-03):
    t_start = time.time()
    factors_sub = []
    
    for mode in range(test_tensor.ndim):
        unfolded = tl.unfold(test_tensor, mode)
        m, n = unfolded.shape
        n_samples = int(n * sampling_ratio)
        sample_indices = np.random.choice(n, size=n_samples, replace=False)
        sampled_matrix = unfolded[:, sample_indices]
        
        U, S, _ = np.linalg.svd(sampled_matrix, full_matrices=False)
        
        # Truncate based on threshold: keep singular values >= sv_threshold * max(S)
        threshold = sv_threshold * S[0]  # S[0] is the maximum singular value
        rank = np.sum(S >= threshold)
        rank = max(1, rank)  # Ensure at least one singular value is kept
        
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

def low_cost_hosvd(test_tensor, time_it=False, sv_threshold=1e-03, reduction_factor=2):
    t_start = time.time()
    subsampled_tensor_lc = test_tensor[:, :, ::reduction_factor]
    factors_lc = []
    
    for mode in range(test_tensor.ndim):
        unfolded = tl.unfold(subsampled_tensor_lc, mode)
        
        if mode != test_tensor.ndim - 1:
            U, S, _ = np.linalg.svd(unfolded, full_matrices=False)
            
            # Truncate based on threshold
            threshold = sv_threshold * S[0]
            rank = np.sum(S >= threshold)
            rank = max(1, rank)
            
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
            
            # Truncate based on threshold
            threshold = sv_threshold * S[0]
            rank = np.sum(S >= threshold)
            rank = max(1, rank)
            
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
    return tl.tenalg.multi_mode_dot(core, factors, modes=[i for i in range(len(factors))])

def compute_error(test_tensor, reconstruction):
    return np.linalg.norm(np.subtract(test_tensor, reconstruction)) / np.linalg.norm(test_tensor)

def compute_compression_factor(original_tensor, core, factors):
    original_size = original_tensor.size
    decomp_size = core.size + sum(U.size for U in factors)
    return original_size / decomp_size