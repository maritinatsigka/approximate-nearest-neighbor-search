#pragma once
#include <vector>

class IVFFlat {
public:
    IVFFlat(int _n, int _dim, int _kclusters, int _nprobe, unsigned int _seed);

    void Build_index(const std::vector<std::vector<double>>& dataset);
    std::vector<int> ANN(const std::vector<double>& q, int numNeighbors);
    std::vector<int> Range(const std::vector<double>& q, double R);
    
private:
    //parameters.
    int n, dim, kclusters, nprobe; 
    unsigned int seed;

    int max_iters; //nunmber of lloyd runs.
    double tol; //when to stop (centroids stop moving).

    const std::vector<std::vector<double>>* pdata; //dataset poijtner.

    // k-means state
    std::vector<std::vector<double>> centroids;
    std::vector<int> assignment; //which cluster each point belongs to.

    // Inverted lists.
    std::vector<std::vector<int>> invlists;

    //helpers.
    void kmeans(); //Lloyd loop.
    void kmeans_init_pp();  //k-means++ init.
    void rebuild_lists(); //Rebuild inverted lists after k-means.
};