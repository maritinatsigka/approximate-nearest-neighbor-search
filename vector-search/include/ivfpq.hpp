#pragma once
#include <vector>
#include <cstdint>
#include <utility>

//IVF+PQ index: coarse k-means lists + product quantization codes
class IVFPQ {
public:
    IVFPQ(int _n, int _dim, int _kclusters, int _nprobe, int _M, int _nbits, unsigned int _seed);
    void Build_index(const std::vector<std::vector<double>>& dataset);  //train + encode + build lists
    std::vector<int> ANN(const std::vector<double>& q, int numNeighbors);  //approx top-N
    std::vector<int> Range(const std::vector<double>& q, double R);   //ids within radius R

private:
    int n, dim;
    int kclusters;  //coarse vocab size (IVF)
    int nprobe;     //how many coarse lists to scan
    int M;         //#subspaces for PQ
    int bits;     // bits per subspace code
    int Ks;      //codebook size per subspace (2^bits)
    unsigned int seed;
    int max_iters;  //k-means iters
    double tol;   //stop threshold

    //data & PQ layout
    const std::vector<std::vector<double>>* data = nullptr;
    std::vector<std::vector<int>> sub;   //round-robin subspace indices [M][dm]
    std::vector<std::vector<std::vector<double>>> cb;   //codebooks [M][Ks][dm]

    //coarse quantizer and inverted lists
    std::vector<std::vector<double>> centroids;  //coarse centroids [kclusters][dim]
    std::vector<int> assignment;           //point -> coarse id
    std::vector<std::vector<int>> lists;      //inverted lists: coarse id-> vector ids
    std::vector<std::vector<std::vector<uint16_t>>> codes;  //per-vector PQ codes [coarse][idx][m]

    //subspace layout
    void build_roundrobin_layout();
    inline int subspace_dim(int m) const { return (int)sub[m].size(); }

    //coarse (IVF)
    void train_coarse();  //k-means on full dim
    void assign_coarse();  //point -> nearest centroid
    static void kmeans_forgy(const std::vector<std::vector<double>>& X, int K, int max_iters, double tol, unsigned int seed, std::vector<std::vector<double>>& C, std::vector<int>& assign);

    //PQ training/encoding
    void train_pq_codebooks();  //k-means per subspace
    void encode_vector(const std::vector<double>& x, std::vector<uint16_t>& out_codes) const;  //argmin per subspace

    //distance helpers (round-robin)
    double l2_sq_subspace_q_centroid(int m, const std::vector<double>& q, const std::vector<double>& c) const;
    double l2_sq_full(const std::vector<double>& a, const std::vector<double>& b) const; // wrapper of l2_sq

    //ANN pipeline
    std::vector<int> shortlist_centroids(const std::vector<double>& q, int b) const;   //top-b coarse
    void build_distance_table(const std::vector<double>& q, std::vector<std::vector<double>>& distab) const; //distab[m][k]
    std::vector<std::pair<double,int>> gather_candidates(const std::vector<int>& cids, const std::vector<std::vector<double>>& distab) const;  //ADC sums
    std::vector<int> select_topN(std::vector<std::pair<double,int>>& dist_id, int N) const;   //pick best ids
};
