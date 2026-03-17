#include <random>
#include <algorithm>
#include <limits>
#include <numeric>
#include <iostream>
#include <cmath>
#include "ivfflat.hpp"
#include "utils.hpp"
using namespace std;

//default values for our params
IVFFlat::IVFFlat(int _n, int _dim, int _kclusters, int _nprobe, unsigned int _seed)
    : n(_n), dim(_dim), kclusters(_kclusters > 1 ? _kclusters : 50), nprobe(_nprobe > 0 ? _nprobe : 5), seed(_seed ? _seed : 1), max_iters(50), tol(1e-4), pdata(nullptr)
{
    //make room for the centroids and for cluster assignments.
    centroids.assign(kclusters, vector<double>(dim, 0.0));
    assignment.assign(n, -1);
}

//Build the IVF index.
void IVFFlat::Build_index(const vector<vector<double>>& dataset){
    pdata = &dataset; //pointer to the dataset.

    if (!check_consistent_dim(dataset)){
        std::cerr << "IVFFlat::Build_index: inconsistent vector dimensions\n";
        return;
    }
    
    if (pdata->empty()) return; // nothing to do.

    n   = (int)pdata->size();
    dim = (int)(*pdata)[0].size();
    //avoid cases when k > n.
    if (kclusters > n) kclusters = n;
    // reallocate space in case dimensions changed.
    centroids.assign(kclusters, vector<double>(dim, 0.0));
    assignment.assign(n, -1);
    invlists.assign(kclusters, {});

    kmeans(); // run clustering
    rebuild_lists(); //after k-means put each point into its cluster list
}

// k-means++ INIT.
void IVFFlat::kmeans_init_pp(){
    mt19937_64 gen(seed);
    uniform_int_distribution<int> pick0(0, n-1);

    //pick first a random centroid from the dataset.
    centroids[0] = (*pdata)[pick0(gen)];

    // D[i].Squared dist from point i to the nearest chosen centroid so far.
    vector<double> D(n, 0.0);
    // lambda to refresh D values.
    auto refresh_D = [&](int upto){
        for (int i = 0; i < n; ++i){
            double best = numeric_limits<double>::infinity();
            for (int j = 0; j < upto; ++j){
                double d2 = l2_sq((*pdata)[i], centroids[j]);
                if (d2 < best) best = d2;
            }
            D[i] = best;
        }
    };
    refresh_D(1);
    for (int c = 1; c < kclusters; ++c){
        double sum = accumulate(D.begin(), D.end(), 0.0);
        //IF all points are the same, just pick random.
        if (sum <= 0.0){
            centroids[c] = (*pdata)[pick0(gen)];
            refresh_D(c+1);
            continue;
        }
        // pick a new centroid weighted by D^2. 
        uniform_real_distribution<double> unif(0.0, sum);
        double r = unif(gen);

        int chosen = 0;
        double acc = 0.0;
        for (int i = 0; i < n; ++i){
            acc += D[i];
            if (acc >= r){ chosen = i; break; }
        }
        centroids[c] = (*pdata)[chosen];
        
        //Update D with the new centroid (only gets smaller).
        for (int i = 0; i < n; ++i){
            double d2 = l2_sq((*pdata)[i], centroids[c]);
            if (d2 < D[i]) D[i] = d2;
        }
    }
}

// Lloyd's k-means.
void IVFFlat::kmeans(){
    if (!pdata || pdata->empty()) return;

    //Better starting points, basically fewer iters.
    kmeans_init_pp();

    vector<int> counts(kclusters, 0); // points per cluster
    vector<vector<double>> sums(kclusters, vector<double>(dim, 0.0)); //Sum of points per cluster
    double prev_shift = numeric_limits<double>::infinity();

    for (int it = 0; it < max_iters; ++it){
        // Assignment step.
        for (int i = 0; i < n; ++i){
            int bestJ = 0;
            double bestD = numeric_limits<double>::infinity();
            for (int j = 0; j < kclusters; ++j){
                double d2 = l2_sq((*pdata)[i], centroids[j]);
                if (d2 < bestD){ bestD = d2; bestJ = j; }
            }
            assignment[i] = bestJ;
        }

        //reset accumulators for update step.
        for (int j = 0; j < kclusters; ++j){
            fill(sums[j].begin(), sums[j].end(), 0.0);
            counts[j] = 0;
        }

        //Accumulate sums per cluster.
        for (int i = 0; i < n; ++i){
            int j = assignment[i];
            counts[j]++;
            const auto& v = (*pdata)[i];
            auto& s = sums[j];
            for (int d = 0; d < dim; ++d) s[d] += v[d];
        }

        // If a cluster is empty, just re-seed it a random point.
        mt19937_64 gen(seed + it + 123);
        uniform_int_distribution<int> uni(0, n-1);

        double total_shift = 0.0; // total centroid moved.
        for (int j = 0; j < kclusters; ++j){
            vector<double> next(dim, 0.0);
            if (counts[j] == 0){
                //re-seed to a random point.
                next = (*pdata)[uni(gen)];
            } else {
                for (int d = 0; d < dim; ++d) next[d] = sums[j][d] / counts[j];
            }
            //We measure movement with true L2
            total_shift += l2(centroids[j], next);
            centroids[j].swap(next);
        }

        //If movement barely changes, we stop.
        if (fabs(total_shift - prev_shift) < tol) break;
        prev_shift = total_shift;
    }
}

// After k-means Rebuild the inverted lists.
void IVFFlat::rebuild_lists(){
    invlists.assign(kclusters, {}); //Clear old lists.
    for (int i = 0; i < n; ++i){
        int j = assignment[i]; // which cluster this point belongs to
        if (j >= 0 && j < kclusters) invlists[j].push_back(i); //add point to the right cluster list
    }
}

// ANN search.
vector<int> IVFFlat::ANN(const vector<double>& q, int numNeighbors){
    
    if (!pdata || (int)q.size() != dim || numNeighbors <= 0) return {};
    vector<int> empty;
    if (!pdata) return empty;

    //Find the nprobe closest centroids.
    int b = nprobe;
    if (b < 1) b = 1;
    if (b > kclusters) b = kclusters;
    // compute distances from query to all centroids
    vector<pair<double,int>> cd(kclusters);
    for (int j = 0; j < kclusters; ++j){
        cd[j] = { l2_sq(q, centroids[j]), j };
    }

    // Get the closest b centroids.
    if (b < kclusters){
        nth_element(cd.begin(), cd.begin()+b, cd.end());
        cd.resize(b);
    } else {
        sort(cd.begin(), cd.end());
    }

    // Collect candidates from the selected lists.
    vector<pair<double,int>> dist_id;
    dist_id.reserve(128); //SMALL reserve to reduce reallocations.
    for (auto &p : cd){
        int j = p.second;
        const auto& lst = invlists[j];
        for (int id : lst){
            // compute real distance from query to the point
            double d2 = l2_sq(q, (*pdata)[id]);
            dist_id.emplace_back(d2, id);
        }
    }

    //select nearest neighbors among candidates.
    if ((int)dist_id.size() > numNeighbors){
        partial_sort(dist_id.begin(), dist_id.begin()+numNeighbors, dist_id.end());
        dist_id.resize(numNeighbors);
    } else {
        sort(dist_id.begin(), dist_id.end());
    }

    // Prepare result, only the ids of the nearest neighbors.
    vector<int> result;
    result.reserve(dist_id.size());
    for (auto &pr : dist_id) result.push_back(pr.second);
    return result;
}

// Range search for IVFLAT.
vector<int> IVFFlat::Range(const vector<double>& q, double R){
    
    if (!pdata || (int)q.size() != dim || R < 0.0) return {};
    vector<int> ids;

    //Find the nprobe closest centroids.
    int b = nprobe;
    if (b < 1) b = 1;
    if (b > kclusters) b = kclusters;

    // Use squared radius so we avoid taking sqrt() every time
    double R2 = R * R;
    //Compute distances of query to all centroids
    vector<pair<double,int>> cd(kclusters);
    for (int j = 0; j < kclusters; ++j){
        cd[j] = { l2_sq(q, centroids[j]), j };
    }
    // keep only the closest b centroids to probe
    if (b < kclusters){
        nth_element(cd.begin(), cd.begin()+b, cd.end());
        cd.resize(b);
    } else {
        sort(cd.begin(), cd.end());
    }

    // Collect candidates from the selected lists.
    for (auto &p : cd){
        int j = p.second;
        const auto& lst = invlists[j];
        for (int id : lst){
            //Compare squared distances with R^2 for speed.
            if (l2_sq(q, (*pdata)[id]) <= R2) ids.push_back(id);
        }
    }
    return ids; // return all points within radius R
}
