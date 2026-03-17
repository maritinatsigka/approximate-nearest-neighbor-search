#include "LSH.hpp"
#include "utils.hpp"
#include <random>
#include <cmath>
#include <iostream>
#include <algorithm>
using namespace std;

// LSH constructor.
LSH::LSH(int _dim, int _k, int _L, double _w, unsigned int _seed)
    : dim(_dim), k(_k), L(_L), w(_w <= 0.0 ? 4.0 : _w), seed(_seed ? _seed : 1), pdata(nullptr){
    mt19937_64 gen(seed);
    normal_distribution<double> nd(0.0, 1.0); //gaussian for projections
    uniform_real_distribution<double> ud(0.0, w); //uniform [0, w) for shifts

    //allocate projections (v) and shifts (t).
    vs.assign(L, vector<vector<double>>(k, vector<double>(dim)));
    ts.assign(L, vector<double>(k));
    tables.assign(L, {}); 
    // Fill v and t with random values
    for (int i = 0; i<L; i++) {
        for (int j = 0; j < k;j++) {
            for (int d = 0; d < dim; d++) {
                vs[i][j][d] = nd(gen);
            }
            ts[i][j] = ud(gen);
        }
    }
    //Random integers for hash combination
    r.assign(L, vector<int>(k));
    uniform_int_distribution<int> rdist(1, (1 << 30));
    for (int i = 0; i < L; ++i)
        for (int j = 0; j < k; ++j) r[i][j] = rdist(gen);

    M = 4294967291u; //large 32bit prime for hashing
    tableSize = 0; //Set during Build_index
}

// h function.
long long LSH::h(const vector<double>& p, const vector<double>& vj, double tj) const
{
    double dot = 0.0;
    for (int i = 0; i < dim; i++) {
        dot += p[i] * vj[i]; //dot product between p and vj
    }
    long long hh = (long long) floor((dot + tj) / w);
    return hh;
}
// g function multiple h() values into one hash index.
long long LSH::g(const vector<double>& p, int i) const{
    unsigned long long id = (unsigned long long)ID(p, i);
    return (long long)(id % tableSize);//map to table size
}


// Put each point into each hash table bucket.
void LSH::Build_index(const vector<vector<double>>& dataset) {
    // Set pointer to dataset. 
    pdata = &dataset;

    if (!check_consistent_dim(dataset)){
        std::cerr << "LSH::Build_index: inconsistent vector dimensions\n";
        return;
    }
    
    if (dataset.empty()) return;
    //choose table size based on dataset size and dimension.
    const size_t n = dataset.size();
    size_t div = (size_t)(dim <= 128 ? 16 : 4);
    tableSize = max<size_t>(256, n / div);

    // Reinitialize tables for a fresh build.
    tables.clear();
    tables.resize(L);
    //Insert each point into each hash table.
    for (int i = 0; i < L; ++i) {
        for (int id = 0; id < (int)dataset.size(); ++id) {
            long long gv = g(dataset[id], i);
            tables[i][gv].push_back(id); //append point id to that bucket
        }
    }
}

// ID function combines k hash values into one number for table i.
long long LSH::ID(const vector<double>& p, int i) const {
    unsigned long long sum = 0ULL;
    for (int j = 0; j < k; j++) {
        //get hash value for this projection.
        long long hj = h(p, vs[i][j], ts[i][j]);
        unsigned long long term =
            ((unsigned long long) r[i][j] * (unsigned long long) (hj & 0xffffffffULL)) % M;
        sum += term;
        if (sum >= M) sum -= M; //prevent overflow
    }
    //return final combined hash ID.
    return (long long) (sum % M);
}
// ANN function 
vector<int> LSH::ANN(const vector<double>& q, int numNeighbors)
{
    if (!pdata || tableSize == 0 || (int)q.size() != dim || numNeighbors <= 0) return {};
    //Candidate points.
    vector<int> candidates;

    // take candidates from hash table.
    int pulled = 0;
    const int MAXC = 200 * L; //max candidates to consider.
    const int OFFS[5] = {0, +1, -1, +2, -2};
    //For each table Get the bucket and add points to candidates.
    for (int i = 0; i < L; ++i) {
        //base bucket for this table.
        unsigned long long idq = (unsigned long long)ID(q, i);
        long long base = (long long)(idq % tableSize);

        //Pull from base bucket and its neighbors.
        for (int oi = 0; oi < 5 && pulled < MAXC; ++oi) {
            long long gq = base + OFFS[oi];
            // wrap around table size
            if (gq < 0) gq += (long long)tableSize;
            if ((unsigned long long)gq >= tableSize) gq -= (long long)tableSize;
            auto it = tables[i].find(gq);
            if (it != tables[i].end()) {
                for (int id : it->second) {
                    candidates.push_back(id);
                    if (++pulled >= MAXC) break;
                }
            }
        }
        if (pulled >= MAXC) break;
    }
    // Remove duplicates.
    sort(candidates.begin(), candidates.end());
    candidates.erase(unique(candidates.begin(), candidates.end()), candidates.end());
    if (candidates.empty()) return candidates;

    //distances and select nearest neighbors.
    vector<pair<double,int>> scored;
    scored.reserve(candidates.size());
    for (int id : candidates) {
        double d2 = l2_sq(q, (*pdata)[id]);
        scored.emplace_back(d2, id);
    }
    // keep the best N (partial selection, then sort the top-N).
    if ((int)scored.size() > numNeighbors) {
        nth_element(scored.begin(), scored.begin() + numNeighbors, scored.end());
        sort(scored.begin(), scored.begin() + numNeighbors);
        scored.resize(numNeighbors);
    } else {
        sort(scored.begin(), scored.end());
    }

    //Return only the ids.
    vector<int> result;
    result.reserve(scored.size());
    for (auto &pr : scored) result.push_back(pr.second);
    return result;
}