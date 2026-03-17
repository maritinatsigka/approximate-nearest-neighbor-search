#pragma once
#include <vector>
#include <unordered_map>
#include <cstdint>

//binary hypercube index (random projections -> bits -> vertex buckets)
class Hypercube{
public:
    Hypercube(int count, int dim, int kproj, double w, int max_check, int max_probes, unsigned int seed = 1);
    void BuildIndex(const std::vector<std::vector<double>>& data);  //build buckets
    std::vector<int> ANN(const std::vector<double>& q, int N) const;  //top-N ids
    std::vector<int> Range(const std::vector<double>& q, double R) const;   //ids within R (L2)

private:
    //params
    int n, d, k, M, probes;
    double width;
    unsigned int rng_seed;

    //data + hash storage
    const std::vector<std::vector<double>>* data_ptr = nullptr;
    std::vector<std::vector<double>> proj; //k Gaussian projection vectors
    std::vector<double> shift;   //k random shifts in [0,w)
    std::unordered_map<uint64_t, std::vector<int>> table;  //vertex -> point ids

    mutable std::vector<std::unordered_map<long long, int>> bit_cache;

    //init & hashing pieces
    void init_random();    
    double proj_dot(const std::vector<double>& p, int j) const;

    long long bucket(const std::vector<double>& p, int j) const; 
    int bit_of(const std::vector<double>& p, int j) const;   //stable 0/1 from bucket
    uint64_t vertex(const std::vector<double>& p) const;   //k-bit id

    //probe generation (increasing Hamming distance)
    void make_probes(uint64_t start, int max_probes, std::vector<uint64_t>& out) const; 

    //candidate collection & filtering
    std::vector<int> collect_candidates(const std::vector<uint64_t>& vids) const;
    void dedup_inplace(std::vector<int>& v) const;
    std::vector<int> score_topN(const std::vector<double>& q, const std::vector<int>& cand, int N) const;
    std::vector<int> filter_range(const std::vector<double>& q, const std::vector<int>& cand, double R) const;
};
