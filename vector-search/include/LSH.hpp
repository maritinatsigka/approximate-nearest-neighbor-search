#pragma once
#include <vector>
#include <unordered_map>
#include <cstdint> 
#include <cstddef>

class LSH {
public:
    LSH(int dim, int k, int L, double w, unsigned int seed = 1);

    // Build and query
    void Build_index(const std::vector<std::vector<double>>& dataset);
    std::vector<int> ANN(const std::vector<double>& q, int numNeighbors);

private:
    // Hash pieces.
    long long g(const std::vector<double>& p, int tableindex) const;
    long long h(const std::vector<double>& p, const std::vector<double>& vj, double tj) const;
    long long ID(const std::vector<double>& p, int i) const;  

    // params.
    int dim, k, L;
    double w;
    unsigned int seed;

    // Projections and shifts.
    std::vector<std::vector<std::vector<double>>> vs; // [L][k][dim]
    std::vector<std::vector<double>> ts; // [L][k]

    // Amplification multipliers per table.
    std::vector<std::vector<int>> r; // [L][k]

    // Hash tables.
    std::vector<std::unordered_map<long long, std::vector<int>>> tables;

    // Data pointer and indexing.
    const std::vector<std::vector<double>>* pdata = nullptr;
    uint32_t M = 0;
    std::size_t tableSize = 0;
};