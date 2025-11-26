# Low Cost HOSVD for Combustion Data
I will make this file to showcase the issue with low cost SVD in combustion, I will make it as clear as possible at the risk of writing redundant stuff.

## Classical SVD
This section is for nomenclature

### Definition
For a matrix $\mathbf{X} \in \mathbb{R}^{m \times n}$, the Singular Value Decomposition (SVD) is defined as:

$$\mathbf{X} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$$

where:
- $\mathbf{U} \in \mathbb{R}^{m \times m}$ is an orthogonal matrix containing the left singular vectors
- $\mathbf{\Sigma} \in \mathbb{R}^{m \times n}$ is a diagonal matrix containing the singular values $\sigma_1 \geq \sigma_2 \geq ... \geq \sigma_r > 0$ (where $r = \text{rank}(\mathbf{X})$)
- $\mathbf{V} \in \mathbb{R}^{n \times n}$ is an orthogonal matrix containing the right singular vectors

The left singular vectors in $\mathbf{U}$ represent the orthogonal basis for the row space of $\mathbf{X}$, capturing patterns across the $m$ rows. The right singular vectors in $\mathbf{V}$ represent the orthogonal basis for the column space of $\mathbf{X}$, capturing patterns across the $n$ columns. The singular values in $\mathbf{\Sigma}$ quantify the importance or energy associated with each pair of singular vectors, ordered from most to least significant.

### Low-Rank Approximation
The truncated SVD provides the best rank-$k$ approximation to $\mathbf{X}$ in the least squares sense:

$$\mathbf{X}_k = \mathbf{U}_k \mathbf{\Sigma}_k \mathbf{V}_k^T = \sum_{i=1}^k \sigma_i \mathbf{u}_i \mathbf{v}_i^T$$

where $\mathbf{U}_k \in \mathbb{R}^{m \times k}$, $\mathbf{\Sigma}_k \in \mathbb{R}^{k \times k}$, and $\mathbf{V}_k \in \mathbb{R}^{n \times k}$ contain the first $k$ singular vectors and values.

The approximation error is:
$$\|\mathbf{X} - \mathbf{X}_k\|_F = \sqrt{\sum_{i=k+1}^r \sigma_i^2}$$

## Low Cost SVD

### Motivation

For large matrices where $n$ (number of columns) is very large, computing the full SVD is very expensive. The Low Cost SVD algorithm has as its purpose to recover a good approximation of all three matrices $\mathbf{U}$, $\mathbf{\Sigma}$, and $\mathbf{V}$. We consider the case of **short fat matrices**, i.e., $\mathbf{X} \in \mathbb{R}^{m \times n}$ with $m \ll n$. The first step is to subsample in the column space. Note that if we had a tall skinny matrix ($m \gg n$), as in the original paper, the discussion would be the same but the focus would change from using the algorithm to estimate to estimate the row matrix ($\mathbf{U}$) instead of the column matrix $\mathbf{V}$. For the sake of consistency with the HOSVD formulation, both the one from modelflows code and the one in the original paper, we discuss short fat matrices. Two of the three matrices come "for free" in the sense that, as a hypothesis of the algorithm, if we subsample in the column space and $n \gg m$, then $\mathbf{U}$ and $\mathbf{\Sigma}$ are considered almost unvaried: $\mathbf{U}_s \approx \mathbf{U}$ and $\mathbf{\Sigma}_s \approx \mathbf{\Sigma}$, where $\mathbf{U}_s$ and $\mathbf{\Sigma}_s$ are obtained from the SVD of the subsampled matrix $\mathbf{X}_s \in \mathbb{R}^{m \times n_s}$ with $n_s \ll n$.

To be even clearearer, the goal of low cost SVD is to get a good estimate of the matrix concerning the orthogonal basis of the subsampled space from the singular values and the matrix concerning the orthogonal base for the unsampled space. Again, for the case we are considering, this means recovering ($\mathbf{V}$) with ($\mathbf{U}$) and ($\mathbf{\Sigma}$)

### Algorithm
Given a matrix $\mathbf{X} \in \mathbb{R}^{m \times n}$ and a subsampling ratio $s < 1$, the Low Cost SVD proceeds as follows:

1. **Column Subsampling**: Select $n_s = \lfloor s \cdot n \rfloor$ columns from $\mathbf{X}$ to form a reduced matrix $\mathbf{X}_s \in \mathbb{R}^{m \times n_s}$

2. **SVD on Subsampled Matrix**: Compute the SVD of the subsampled matrix:
   $$\mathbf{X}_s = \mathbf{U}_s \mathbf{\Sigma}_s \mathbf{V}_s^T$$

3. **Extract Left Singular Vectors**: Use $\mathbf{U}_s \in \mathbb{R}^{m \times m}$ (or its truncated version $\mathbf{U}_{s,k} \in \mathbb{R}^{m \times k}$) as an approximation to the left singular vectors of the full matrix $\mathbf{X}$

4. **Recover Right Singular Vectors**: Then one can recover a good approximation of the right singular vectors by substituting $\mathbf{X}_s$ with $\mathbf{X}$:
   $$\mathbf{U}_s \mathbf{\Sigma}_s \mathbf{V}_s^T \approx \mathbf{X} \rightarrow \mathbf{V}^T \approx \mathbf{\Sigma}_s^{-1}\mathbf{U}_s^T\mathbf{X}$$
   
   Note the dimensions: $\mathbf{\Sigma}_s^{-1} \in \mathbb{R}^{m \times m}$, $\mathbf{U}_s^T \in \mathbb{R}^{m \times m}$, and $\mathbf{X} \in \mathbb{R}^{m \times n}$, yielding $\mathbf{V}^T \in \mathbb{R}^{m \times n}$.

In this way a good approximation of the right singular vectors is found without having to compute the SVD of the large matrix.

### Sidenote

If one tries to recover the right singular vectors **after having subsampled the columns**, the dimensions do not match (and it would make little to no sense). Specifically:
- From the subsampled SVD, we have $\mathbf{V}_s^T \in \mathbb{R}^{n_s \times n_s}$
- The full matrix requires $\mathbf{V}^T \in \mathbb{R}^{n \times n}$ (or at least $\mathbf{V}^T \in \mathbb{R}^{r \times n}$ where $r = \text{rank}(\mathbf{X})$)
- When attempting the reconstruction $\mathbf{V}^T \approx \mathbf{\Sigma}_s^{-1}\mathbf{U}_s^T\mathbf{X}$, we get $\mathbf{V}^T \in \mathbb{R}^{m \times n}$, but this should have $n$ rows corresponding to all columns, not $m$ rows

The problem is that $\mathbf{V}_s$ only contains information about the $n_s$ subsampled columns, not the full $n$ columns of the original matrix.

## Low Cost SVD on HOSVD

### Higher-Order Singular Value Decomposition (HOSVD)
For a tensor $\mathcal{X} \in \mathbb{R}^{n_1 \times n_2 \times n_3 \times ... \times n_n}$, the HOSVD decomposes the tensor as:

$$\mathcal{X} = \mathcal{S} \times_1 \mathbf{U}^{(1)} \times_2 \mathbf{U}^{(2)} \times_3 \mathbf{U}^{(3)} ... \times_n \mathbf{U}^{(n)}$$

where:
- $\mathcal{S} \in \mathbb{R}^{n_1 \times n_2 \times n_3 \times ... \times n_n}$ is the core tensor
- $\mathbf{U}^{(i)} \in \mathbb{R}^{n_i \times n_i}$ are orthogonal mode matrices for each dimension $i$
- $\times_i$ denotes the mode-$i$ product

The HOSVD algorithm proceeds as follows:

1. **Mode Unfolding**: For each mode $i = 1, 2, ..., d$, unfold the tensor $\mathcal{X}$ into a matrix $\mathbf{X}_{(i)} \in \mathbb{R}^{n_i \times \prod_{j \neq i} n_j}$

2. **SVD per Mode**: Compute the SVD of each unfolded matrix:
   $$\mathbf{X}_{(i)} = \mathbf{U}^{(i)} \mathbf{\Sigma}^{(i)} \mathbf{V}^{(i)T}$$

3. **Extract Mode Matrices**: Keep only $\mathbf{U}^{(i)}$ (or its truncated version) for each mode

4. **Compute Core Tensor**: The core tensor is obtained by:
   $$\mathcal{S} = \mathcal{X} \times_1 \mathbf{U}^{(1)T} \times_2 \mathbf{U}^{(2)T} \times_3 \mathbf{U}^{(3)T} \times_d ... \times_n \mathbf{U}^{(n)T}$$

### Main issue with low cost SVD

The HOSVD only requires the left singular vectors $\mathbf{U}^{(i)}$ from each mode unfolding (considering ulfolding as short fat matrices). The $\mathbf{V}$ matrix is **not used** in the HOSVD algorithm. . The right singular vectors $\mathbf{V}^{(i)}$ are discarded and play no role in the tensor decomposition or reconstruction, since their information is contained in the core tensor. 
The work I did last week is to understand which ways there are to find good $\mathbf{U}$ by reducing cost (paper sent by mail).

Therefore, the purpose of Low Cost SVD which would be to efficiently recover $\mathbf{V}^{(i)}$ is not useful for HOSVD, as only $\mathbf{U}^{(i)}$ is needed.

