## Introduction

Course: Software Development for Algorithmic Problems  
Winter Semester 2025–2026

In this project, we implemented and evaluated multiple algorithms for Approximate Nearest Neighbor (ANN) search in high-dimensional vector spaces. The goal was to explore efficient alternatives to exact nearest neighbor search, reducing computational cost while maintaining high accuracy.

The algorithms studied in this project are:

- **LSH**  
- **Hypercube**  
- **IVFFlat**  
- **IVFPQ**

Each method supports both **Top-N nearest neighbor search** and **range search within a radius R** around the query.

The experiments were conducted on the well-known datasets **MNIST** (handwritten digit images, 784-dimensional vectors) and **SIFT1M** (128-dimensional feature vectors).

For each algorithm, the following metrics were evaluated:

- **Average Approximation Factor (AF)**,  
- **Recall@N**,  
- **Queries Per Second (QPS)**,
- Average search times (approximate and exact)

Finally, the results were compared in order to analyze how different parameter settings affect performance and to identify the configurations that provide the best balance between speed and accuracy.

---

## Dataset Format

The program expects datasets in the following binary formats:

- **MNIST**: Big-endian binary format (28×28 images flattened to 784-dimensional vectors).
- **SIFT1M**: Little-endian `.fvecs` format (128-dimensional float vectors).

Datasets are not included in this repository and must be downloaded separately.

---

## Output Format

For each query, the program reports:

- Approximate nearest neighbors
- Exact nearest neighbors (for evaluation)
- Distances
- Average Approximation Factor (AF)
- Recall@N
- Queries Per Second (QPS)
- Average approximate and exact search times

---

## Code Structure

```text
include/
├── cli.hpp            # CLI definitions (flag and argument parsing)
├── dataset.hpp        # Dataset loading (MNIST/SIFT)
├── hypercube.hpp      # Hypercube declarations
├── LSH.hpp            # LSH declarations
├── ivfflat.hpp        # IVFFlat declarations
├── ivfpq.hpp          # IVFPQ declarations
└── utils.hpp          # Utility functions

src/
├── dataset.cpp        # Dataset loading implementation
├── hypercube.cpp      # Hypercube main implementation
├── LSH.cpp            # LSH main implementation
├── ivfflat.cpp        # IVFFlat implementation (build/ANN/Range)
├── ivfpq.cpp          # IVFPQ implementation
├── utils.cpp          # Utility functions implementation
└── main.cpp           # Program entry point

outputs/
├── out_mnist_ivfflat.txt
├── out_sift_ivfflat.txt

Makefile               
README.md
```

**Notes**

- The executable file is **`search`**, which is generated using the `make` command.
- Each algorithm is activated via CLI flags: `-lsh`, `-ivfflat`, `-hypercube`, `-ivfpq`.
- Experimental results are stored in the **`outputs/`** directory.
- The codebase is fully modular, with each algorithm implemented in a separate `.cpp/.hpp` pair.
- **Due to the large size of output files, only representative results from the IVFFlat algorithm (on MNIST and SIFT datasets) are included in the `outputs/` directory**.
- **All algorithms (LSH, Hypercube, IVFFlat, IVFPQ) are fully implemented and were evaluated during experimentation**.

## Compilation Instructions

The project is compiled using the provided **Makefile**, which generates the `search` executable.

```bash
cd vector-search
make
```

To remove object files and clean the build:

```bash
make clean
```
---

## Execution Instructions

After compilation, the **`search`** executable can be run from the command line (CLI) using various parameters (flags) that define the selected algorithm and the input/output data.

### LSH Example

```bash
./search \
  -d data_file \
  -q queries_file \
  -type mnist|sift -o outputs/lsh.txt \
  -lsh -k 4 -L 8 -w 4 -N 5 -R 2000 -range false -seed 1
```

### Hypercube Example

```bash
./search \
  -d data_file -q queries_file -type mnist|sift \
  -hypercube -kproj 14 -M 1000 -probes 5 -w 4.0 \
  -N 5 -R 2000 -range false -seed 1 \
  -o outputs/hypercube.txt
```

### IVFFlat Example

```bash
./search \
  -d data_file -q queries_file -type mnist|sift \
  -ivfflat -kclusters 64 -nprobe 10 \
  -N 5 -R 2000 -range false -seed 1 \
  -o outputs/ivf.txt
```

### IVFPQ Example

```bash
./search \
  -d data_file -q queries_file -type mnist|sift \
  -ivfpq -kclusters 128 -nprobe 8 -M 16 -nbits 8 \
  -N 10 -R 2000 -range false -seed 1 \
  -o outputs/ivfpq.txt
```

## Algorithm Description

### LSH

The implementation is based on **Locality Sensitive Hashing (LSH)**, using multiple hash tables and random Gaussian projections.  

The goal is to efficiently approximate nearest neighbors (ANN) in high-dimensional space.

---

### Initialization

The following parameters are used: `dim, k, L, w, seed`.

- If `w <= 0`, it is set to `w = 4.0`.
- If `seed == 0`, it is set to `seed = 1`.

The following structures are generated:

- `vs[L][k][dim]`: random projection vectors drawn from a normal (Gaussian) distribution.
- `ts[L][k]`: random shifts drawn uniformly from the interval [0, w).
- `r[L][k]`: random integers used for the combined hash computation.


A large prime number `M = 4294967291` is used for modular hashing.

---

### Hash Functions

- **h(p, v, t)**: 
   Computes the inner product `dot = p · v` and returns: `floor((dot + t) / w)`.
  
  Each point is therefore mapped to a bucket based on its projection.

- **ID(p, i)**:  

  Combines the `k` hash values of the i-th hash table using random weights `r[i][j]`, followed by modulo `M`.

- **g(p, i)**:  

  Takes `ID(p, i)` and applies modulo `tableSize` to determine the final bucket index in the corresponding hash table.

---

### Index Construction

In `Build_index()`:
- The size of each hash table is defined as (`tableSize = max(256, n / div)`).
- For each data point and each hash table, `g(p, i)` is computed and the point is inserted into the corresponding bucket `tables[i][g]`.

---

### Search (ANN)

1. For each hash table, the bucket of the query is computed along with 4 neighboring buckets (0, ±1, ±2). 
2. Candidate points are collected (up to `MAXC = 200 * L`), and duplicates are removed.
3. Euclidean distances (`l2_sq`) are computed for all candidates.  
4. The `N` closest points are selected and their ids are returned.

---

### Why It Works

- Random Gaussian projections increase the probability that nearby points fall into the same buckets.
- The parameter `k` controls how “tight” the buckets are (larger values ⇒ stricter hashing).  
- The parameter `L` increases the probability of successful retrieval (more tables ⇒ better recall).   
- The parameter `w` controls the coarseness of the hashing.  
- `MAXC` limits the number of candidates in order to maintain low runtime.

---

### Hypercube

The implementation is based on the Random Projection Binary Hypercube approach.  

Each vector is projected onto `kproj` random axes, quantized using width `w`, and mapped to a `kproj`-bit binary code (a vertex of the hypercube).  

Points that share similar bits are mapped to nearby vertices.

---

### Initialization

- Parameters: `n, dim, kproj, w, M, probes, seed`
  - `kproj:` number of random projections (number of bits → hypercube dimension)
  - `w:` quantization width (controls hashing granularity) 
  - `M:` maximum number of candidate points to examine  
  - `probes:` number of neighboring vertices to explore  
  - `seed:` ensures reproducibility  
- The following structures are generated:
  - `proj[kproj][dim]:` random Gaussian projection vectors  
  - `shift[kproj]:` random shifts in the interval [0, w)  
  - `bit_cache[kproj]:` mapping from bucket values to binary bits

---
 
### Index Construction (`BuildIndex`)

In `BuildIndex()`:
- For each point `p`:
  - For each projection `j`, compute `floor((p · proj[j] + shift[j]) / w)` (bucket value).
  - Each bucket is deterministically mapped to a bit `{0,1}` (via cache or hash rule).
  - The `kproj` bits are combined into a vertex id (a `kproj`-bit integer).
- The point id is stored in the corresponding vertex list (hash table structure: `vertex → [point_ids]`).
  
---

### Search (ANN)

1. Compute the vertex corresponding to the query `q`. 
2. Generate a sequence of vertices in increasing Hamming distance from the initial vertex (up to `probes`). 
3. Collect candidates from the corresponding buckets, up to `M` total, removing duplicates.
4. Compute exact Euclidean distances (`l2_sq`) for all candidates. 
5. Return the `N` closest points.
   
---

### Range Search

- The same probing process (steps 1–3) is followed.
- Points satisfying `L2 ≤ R` are returned (in the implementation, comparison with `R²` is used for efficiency).

---

### Why It Works

- Random projections combined with quantization preserve neighborhood structure: nearby points are likely to fall into the same or adjacent vertices of the hypercube.
- `kproj` controls hashing resolution (larger values ⇒ fewer collisions but possible recall loss). 
- `probes` and `M` allow expanding the search around the initial vertex to improve recall at the cost of additional computation.  
- `w` controls quantization granularity: smaller `w` ⇒ more discriminative hashing; larger `w` ⇒ coarser hashing.

### IVFFlat

The implementation is based on the **Inverted File with Flat lists (IVFFlat)** method.  

The dataset is partitioned into clusters using **k-means**, and for each centroid an inverted list stores the ids of assigned points.

During search, only the closest centroids (determined by `nprobe`) are examined, significantly reducing the number of distance computations.

---

### Initialization

- Parameters: `n, dim, kclusters, nprobe, seed`
  - If `kclusters <= 1`, then `kclusters = 50`  
  - If `nprobe <= 0`, then `nprobe = 5`  
  - If `seed == 0`, then `seed = 1`
- Structures:
  - `centroids[kclusters][dim]` — centroid positions  
  - `assignment[n]` — cluster assignment for each point 
  - `invlists[kclusters]` — inverted lists storing point ids per cluster
- k-means convergence constants: `max_iters = 50`, `tol = 1e-4`

---

### Index Construction (`Build_index`)

1. Validate dimensions and update `n`, `dim`.
2. If `kclusters > n`, reduce it to `n`.   
3. Reinitialize `centroids`, `assignment`, and `invlists`. 
4. Run `kmeans()` followed by `rebuild_lists()` to populate the inverted lists.

---

### K-means

- **k-means++ init (`kmeans_init_pp`)**: The first centroid is selected randomly, and the remaining ones are selected with probability proportional to the squared distance (D²).  
- **Lloyd’s iteration**:
  - Assign each point to its nearest centroid (using `l2_sq`). 
  - Update centroids as the mean of their assigned points.  
  - Empty clusters are reseeded using a random point.  
  - The process stops when the total centroid shift falls below `tol` or when `max_iters` is reached.
    
---

### Nearest Neighbor Search (`ANN`)

1. Compute distances from the query `q` to **all** centroids.  
2. Select the `nprobe` closest centroids. 
3. Collect candidate points from the corresponding `invlists`. 
4. Compute exact distances (`l2_sq`). 
5. Return the `numNeighbors` closest points (using `partial_sort` / `sort`).

---

### Why It Works

- The algorithm partitions the dataset into clusters, so nearby points tend to belong to the same or neighboring clusters. 
- Search is therefore restricted to a limited number of regions in the space, significantly reducing the number of candidate comparisons. 
- The parameter `nprobe` ensures that, if the query lies near cluster boundaries, multiple nearby clusters are examined to avoid missing relevant neighbors.
  
---

### IVFPQ

This method combines **Inverted File (IVF)** with **Product Quantization (PQ)**.

First, the space is coarsely quantized using k-means (IVF), grouping points into `kclusters` inverted lists.  

Then, each vector is compactly encoded using PQ by splitting it into `M` subspaces and learning `2^nbits` centroids per subspace.  

This enables fast approximate distance computation during search.

---

### Initialization

- Parameters: `n, dim, kclusters, nprobe, M, nbits, seed` 
  - `nprobe:` number of coarse lists examined per query
  - `M:` number of subspaces (round-robin split of dimensions) 
  - `nbits:` bits per subspace ⇒ codebook size `Ks = 2^nbits` 
  - `seed:` ensures reproducibility 
- Structures:
  - `centroids[kclusters][dim], assignment[n]` (coarse k-means).
  - `sub[M]:` dimension indices per subspace (round-robin split)  
  - `cb[M][Ks][dm]:` PQ codebooks (one per subspace) 
  - `lists[kclusters]:` inverted lists of point ids  
  - `codes[kclusters][…][M]:` stored PQ codes per point 

---

### Index Construction (`Build_index`)

- **Coarse training**: run k-means (Forgy initialization) in the full space → obtain `centroids` and initial assignments. 
- **Coarse assignment**: assign each point to its nearest centroid. 
- **PQ training**: for each subspace `m`, extract sub-vectors and train a codebook with `Ks` centroids. 
- **Encoding**: encode each vector into `M` codes (one per subspace: argmin over the corresponding codebook) and store it in the corresponding `lists[cid]` together with its PQ codes.
---

### Search (ANN)

1. **IVF shortlist**: compute distances from query `q` to all `centroids` and select the `nprobe` closest coarse clusters. 
2. **ADC (Asymmetric Distance Computation)**:
   - Build a distance table `distab[m][k] = ‖q_sub(m) − cb[m][k]‖²` for all subspaces and codebook centroids.
   - For each candidate in the selected lists, sum `distab[m][code_m]` over all `M` subspaces (approximated squared distance).
3. **Selection**: Keep the `N` best candidates (smallest estimated distances) and return their ids.

---


### Range Search

- Only the `nprobe` coarse lists are examined.
- Exact L2 distance is computed in the full space, returning points satisfying `‖q − x‖ ≤ R` (typically using `l2_sq ≤ R²` for efficiency).

---


### Why It Works

- IVF drastically reduces the search space by limiting comparisons to a few coarse clusters.  
- PQ enables extremely fast approximate distance computation using only `M` subspace codes.  
- Parameter effects:
  - Increasing `nprobe` ⇒ higher recall and better approximation factor, at the cost of runtime/QPS. 
  - Increasing `M` and/or `nbits` ⇒ more accurate ADC approximation (higher recall), with moderate additional memory and computational cost.

---
