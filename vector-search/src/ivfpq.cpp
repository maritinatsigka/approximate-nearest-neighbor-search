#include "ivfpq.hpp"
#include "utils.hpp"
#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <limits>
#include <cmath>

using namespace std;

//store params, build subspace layout, allocate centroids/codebooks
IVFPQ::IVFPQ(int _n, int _dim, int _kclusters, int _nprobe, int _M, int _nbits, unsigned int _seed)
    : n(_n), dim(_dim), kclusters(_kclusters > 1 ? _kclusters : 50), nprobe(_nprobe > 0 ? _nprobe : 5), M(_M > 0 ? _M : 16), 
    bits(_nbits > 0 ? _nbits : 8), Ks(1 << (_nbits > 15 ? 15 : _nbits)), seed(_seed ? _seed : 1), max_iters(40), tol(1e-4), data(nullptr){
    
    if (M > dim){  //clamp M to dimensionality
        M = dim;
    }

    build_roundrobin_layout();  //subspace assignment [0...dim-1] -> round-robin to M
    centroids.assign(kclusters, vector<double>(dim, 0.0));
    assignment.assign(max(0, n), -1);

    //allocate PQ codebooks: cb[m][k] is centroid k in subspace m
    cb.assign(M, {});
    for (int m = 0; m < M; ++m){
        int dm = subspace_dim(m);
        cb[m].assign(Ks, vector<double>(dm, 0.0));
    }
}

//round-robin split of dimensions into M subspaces
void IVFPQ::build_roundrobin_layout(){
    sub.assign(M, {});
    for (int j = 0; j < dim; ++j){
        sub[j % M].push_back(j);
    }
}

//wrapper to shared squared L2
double IVFPQ::l2_sq_full(const vector<double>& a, const vector<double>& b) const{
    return l2_sq(a, b); 
}

//squared distance on one subspace
double IVFPQ::l2_sq_subspace_q_centroid(int m, const vector<double>& q, const vector<double>& c) const{
    const auto& idx = sub[m];
    double s = 0.0;
    for (int t = 0; t < (int)idx.size(); ++t){
        double diff = q[idx[t]] - c[t];
        s += diff * diff;
    }
    return s;
}

//k-means on the full space to train IVF centroids
void IVFPQ::train_coarse(){
    vector<int> tmp_assign;
    kmeans_forgy(*data, kclusters, max_iters, tol, seed, centroids, tmp_assign);
    assignment.swap(tmp_assign);
}

//assign each vector to nearest coarse centroid
void IVFPQ::assign_coarse(){
    const int N = (int)data->size();
    assignment.assign(N, -1);

    for (int i = 0; i < N; ++i){
        int bestJ = 0;
        double bestD = numeric_limits<double>::infinity();

        for (int j = 0; j < kclusters; ++j){
            double d2 = l2_sq_full((*data)[i], centroids[j]);
            if (d2 < bestD){
                bestD = d2;
                bestJ = j;
            }
        }

        assignment[i] = bestJ;
    }
}

//simple Forgy k-means (used both for IVF and PQ subspaces)
void IVFPQ::kmeans_forgy(const vector<vector<double>>& X, int K, int max_iters, double tol, unsigned int seed, vector<vector<double>>& C, vector<int>& assign){
    const int N = (int)X.size();
    if (N == 0){
        C.clear();
        assign.clear();
        return;
    }
    const int D = (int)X[0].size();

    mt19937_64 gen(seed);
    vector<int> ids(N);
    iota(ids.begin(), ids.end(), 0);
    shuffle(ids.begin(), ids.end(), gen);
    ids.resize(min(K, N));

    //init K centers from random distinct samples (wrap if K>N)
    C.assign(K, vector<double>(D, 0.0));
    for (int j = 0; j < (int)ids.size(); ++j){
        C[j] = X[ids[j]];
    }
    
    for (int j = (int)ids.size(); j < K; ++j){
        C[j] = X[ids[j % ids.size()]];
    }

    assign.assign(N, -1);

    vector<vector<double>> sums(K, vector<double>(D, 0.0));
    vector<int> counts(K, 0);

    for (int it = 0; it < max_iters; ++it){
        //assignment step
        for (int i = 0; i < N; ++i){
            int bestJ = 0;
            double bestD = numeric_limits<double>::infinity();

            for (int j = 0; j < K; ++j){
                double d2 = 0.0;
                for (int dd = 0; dd < D; ++dd){
                    double diff = X[i][dd] - C[j][dd];
                    d2 += diff * diff;
                }
                if (d2 < bestD){
                    bestD = d2;
                    bestJ = j;
                }
            }

            assign[i] = bestJ;
        }

        //reset accumulators
        for (int j = 0; j < K; ++j){
            fill(sums[j].begin(), sums[j].end(), 0.0);
            counts[j] = 0;
        }

        //accumulate new means
        for (int i = 0; i < N; ++i){
            int cid = assign[i];
            counts[cid]++;
            const auto& xi = X[i];
            auto& sj = sums[cid];
            for (int dd = 0; dd < D; ++dd){
                sj[dd] += xi[dd];
            }
        }

        //update step + total shift
        double shift = 0.0;
        for (int j = 0; j < K; ++j){
            vector<double> nextC(D, 0.0);

            if (counts[j] == 0){   //empty cluster -> re-seed
                nextC = X[ids[j % ids.size()]];
            } else {
                for (int dd = 0; dd < D; ++dd){
                    nextC[dd] = sums[j][dd] / counts[j];
                }
            }

            shift += l2(C[j], nextC);
            C[j].swap(nextC);
        }

        if (shift < tol){
            break;
        }
    }
}

//train PQ codebooks per subspace (Ks centroids each)
void IVFPQ::train_pq_codebooks(){
    const int N = (int)data->size();

    for (int m = 0; m < M; ++m){
        const auto& idx = sub[m];
        const int dm = (int)idx.size();

        //extract subspace features
        vector<vector<double>> Xm(N, vector<double>(dm, 0.0));
        for (int i = 0; i < N; ++i){
            const auto& x = (*data)[i];
            for (int t = 0; t < dm; ++t){
                Xm[i][t] = x[idx[t]];
            }
        }

        //k-means on subspace m
        vector<vector<double>> subC;
        vector<int> tmp_assign;
        kmeans_forgy(Xm, Ks, max_iters, tol, seed + 1337u + m, subC, tmp_assign);

        cb[m].swap(subC);
    }
}

//encode one vector to PQ codes: argmin over Ks per subspace
void IVFPQ::encode_vector(const vector<double>& x, vector<uint16_t>& out_codes) const{
    out_codes.assign(M, 0);

    for (int m = 0; m < M; ++m){
        int bestCode = 0;
        double bestD = numeric_limits<double>::infinity();

        for (int codeId = 0; codeId < Ks; ++codeId){
            double d2 = l2_sq_subspace_q_centroid(m, x, cb[m][codeId]);
            if (d2 < bestD){
                bestD = d2;
                bestCode = codeId;
            }
        }

        out_codes[m] = (uint16_t)bestCode;
    }
}

//build index: layout, train IVF+PQ, encode dataset and fill inverted lists
void IVFPQ::Build_index(const vector<vector<double>>& dataset){
    data = &dataset;

    if (!check_consistent_dim(dataset)){
        std::cerr << "IVFPQ::Build_index: inconsistent vector dimensions\n";
        return;
    }

    if (data->empty()){
        return;
    }

    n = (int)data->size();
    dim = (int)(*data)[0].size();
    if (M > dim){
        M = dim;
    }

    build_roundrobin_layout();
    cb.assign(M, {});
    for (int m = 0; m < M; ++m){
        int dm = subspace_dim(m);
        cb[m].assign(Ks, std::vector<double>(dm, 0.0));
    }

    if (kclusters > n){
        kclusters = n;
    }
    centroids.assign(kclusters, vector<double>(dim, 0.0));

    train_coarse();
    assign_coarse(); 
    train_pq_codebooks();

    //fill inverted lists and store PQ codes per vector
    lists.assign(kclusters, {});
    codes.assign(kclusters, {});

    vector<uint16_t> enc;
    for (int i = 0; i < n; ++i){
        int cid = assignment[i];
        if (cid < 0 || cid >= kclusters){
            continue;
        }
        encode_vector((*data)[i], enc);
        lists[cid].push_back(i);
        codes[cid].push_back(enc);
    }
}

//select b closest coarse centroids to the query
vector<int> IVFPQ::shortlist_centroids(const vector<double>& q, int b) const{
    b = max(1, min(b, kclusters));

    vector<pair<double,int>> dist_to_c(kclusters);
    for (int j = 0; j < kclusters; ++j){
        dist_to_c[j] = { l2_sq_full(q, centroids[j]), j };
    }

    if (b < kclusters){
        nth_element(dist_to_c.begin(), dist_to_c.begin() + b, dist_to_c.end());
        dist_to_c.resize(b);
    } else {
        sort(dist_to_c.begin(), dist_to_c.end());
    }

    vector<int> out;
    out.reserve(b);
    for (auto &p : dist_to_c){
        out.push_back(p.second);
    }
    return out;
}

//precompute ADC distance table
void IVFPQ::build_distance_table(const vector<double>& q, vector<vector<double>>& distab) const{
    distab.assign(M, vector<double>(Ks, 0.0));

    for (int m = 0; m < M; ++m){
        for (int k = 0; k < Ks; ++k){
            distab[m][k] = l2_sq_subspace_q_centroid(m, q, cb[m][k]); // squared απόσταση
        }
    }
}

//accumulate ADC distances per vector from selected coarse lists
vector<pair<double,int>> IVFPQ::gather_candidates(const vector<int>& cids, const vector<vector<double>>& distab) const{
    vector<pair<double,int>> cand;

    for (int cid : cids){
        const auto& id_list   = lists[cid];
        const auto& code_list = codes[cid];

        for (size_t t = 0; t < id_list.size(); ++t){
            const auto& qcodes = code_list[t];

            double sum_sq = 0.0;
            for (int m = 0; m < M; ++m){
                sum_sq += distab[m][ qcodes[m] ];
            }

            cand.emplace_back(sum_sq, id_list[t]);  //keep squared distance (no sqrt)
        }
    }

    return cand;
}

//pick the N smallest (squared) distances and return ids
vector<int> IVFPQ::select_topN(vector<pair<double,int>>& dist_id, int N) const{
    if ((int)dist_id.size() > N){
        partial_sort(dist_id.begin(), dist_id.begin() + N, dist_id.end());
        dist_id.resize(N);
    } else{
        sort(dist_id.begin(), dist_id.end());
    }

    vector<int> out;
    out.reserve(dist_id.size());
    for (auto &p : dist_id){
        out.push_back(p.second);
    }
    return out;
}

//ANN via IVF shortlist + ADC table + top-N selection
vector<int> IVFPQ::ANN(const vector<double>& q, int numNeighbors){
    if (!data || (int)q.size() != dim || numNeighbors <= 0) return {};

    vector<int> empty;
    if (!data){
        return empty;
    }
    auto picked = shortlist_centroids(q, nprobe);

    vector<vector<double>> distab;
    build_distance_table(q, distab);

    auto scored = gather_candidates(picked, distab);
    return select_topN(scored, numNeighbors);
}

//range search: probe IVF lists and filter by exact L2 on full dim
vector<int> IVFPQ::Range(const vector<double>& q, double R){
    if (!data || (int)q.size() != dim || R < 0.0) return {};
    
    vector<int> out;

    auto picked = shortlist_centroids(q, nprobe);
    double R2 = R * R;

    for (int cid : picked){
        const auto& id_list = lists[cid];
        for (int id : id_list){
            if (l2_sq_full(q, (*data)[id]) <= R2){
                out.push_back(id);
            }
        }
    }

    return out;
}
