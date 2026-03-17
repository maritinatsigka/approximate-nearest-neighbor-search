#include "hypercube.hpp"
#include "utils.hpp"
#include <random>
#include <cmath>
#include <algorithm>
#include <iostream>

//store params and initialize random projections/shifts
Hypercube::Hypercube(int count, int dim, int kproj, double w, int max_check, int max_probes, unsigned int seed)
    : n(count), d(dim), k(kproj), M(max_check), probes(max_probes), width(w), rng_seed(seed), data_ptr(nullptr){
    init_random();
}

//build k Gaussian projection vectors and k shifts in [0,w)
void Hypercube::init_random(){
    std::mt19937 gen(rng_seed);
    std::normal_distribution<double> gauss(0.0, 1.0);
    std::uniform_real_distribution<double> uni(0.0, width);

    proj.assign(k, std::vector<double>(d));
    shift.assign(k, 0.0);

    for (int j = 0; j < k; ++j){
        for (int i = 0; i < d; ++i){
            proj[j][i] = gauss(gen);
        }
        shift[j] = uni(gen);
    }

    bit_cache.assign(k, std::unordered_map<long long, int>());
}

//fast dot(p, proj[j])
double Hypercube::proj_dot(const std::vector<double>& p, int j) const{
    double s = 0.0;
    for (int i = 0; i < d; ++i){
        s += p[i] * proj[j][i];
    }
    return s;
}

//quantize projection to integer bucket
long long Hypercube::bucket(const std::vector<double>& p, int j) const{
    double y = (proj_dot(p, j) + shift[j]) / width;
    return static_cast<long long>(std::floor(y));
}

//map bucket to a stable bit {0,1} using a hashed cache
int Hypercube::bit_of(const std::vector<double>& p, int j) const{
    long long h = bucket(p, j);

    auto& mp = bit_cache[j];
    auto it = mp.find(h);
    if (it != mp.end()){
        return it->second;
    }

    //lightweight hash -> single bit
    unsigned long long key = static_cast<unsigned long long>(h) ^ (static_cast<unsigned long long>(j) << 32) ^ static_cast<unsigned long long>(rng_seed);
    std::size_t hv = std::hash<unsigned long long>{}(key);
    int b = static_cast<int>(hv & 1ULL);

    mp[h] = b;
    return b;
}

//pack k bits into a vertex id
uint64_t Hypercube::vertex(const std::vector<double>& p) const{
    uint64_t id = 0ULL;

    for (int j = 0; j < k; ++j){
        id |= (static_cast<uint64_t>(bit_of(p, j)) << j);
    }

    return id;
}

//build the hash table: vertex -> point ids
void Hypercube::BuildIndex(const std::vector<std::vector<double>>& data){
    data_ptr = &data;

    if (!check_consistent_dim(data)){
        std::cerr << "Hypercube::BuildIndex: inconsistent vector dimensions\n";
        return;
    }

    n = static_cast<int>(data.size());
    d = data.empty() ? 0 : static_cast<int>(data[0].size());

    init_random();

    table.clear();
    table.reserve(static_cast<size_t>(std::max(256, n / 8)));

    for (int i = 0; i < static_cast<int>(data.size()); ++i){
        uint64_t v = vertex(data[i]);
        table[v].push_back(i);
    }
}

//generate vertices to probe in increasing Hamming distance
void Hypercube::make_probes(uint64_t start, int max_probes, std::vector<uint64_t>& out) const{
    out.clear();
    out.push_back(start);

    if (max_probes == 1){
        return;
    }

    int produced = 1;

    for (int dist = 1; produced < max_probes && dist <= k; ++dist){
        std::vector<int> idx(dist);

        for (int i = 0; i < dist; i++){
            idx[i] = i;
        }

        auto finished = [&](){ return idx[0] > k - dist; };

        while (!finished() && produced < max_probes){
            uint64_t mask = 0ULL;

            for (int t = 0; t < dist; t++){
                mask |= (1ULL << idx[t]);
            }

            out.push_back(start ^ mask);
            ++produced;

            int p = dist - 1;
            while (p >= 0 && idx[p] == (k - dist + p)){
                --p;
            }
            if (p < 0){
                break;
            }

            ++idx[p];
            for (int q = p + 1; q < dist; ++q){
                idx[q] = idx[q - 1] + 1;
            }
        }
    }
}

//pull up to M candidate ids from the probed vertices
std::vector<int> Hypercube::collect_candidates(const std::vector<uint64_t>& vids) const{
    std::vector<int> cand;
    if (M <= 0){
        return cand;
    }
    
    int seen = 0;

    for (uint64_t v : vids){
        auto it = table.find(v);
        if (it == table.end()){
            continue;
        }

        for (int id : it->second){
            cand.push_back(id);

            if (++seen >= M){
                break;
            }
        }

        if (seen >= M){
            break;
        }
    }
    return cand;
}

//in-place dedup + sort
void Hypercube::dedup_inplace(std::vector<int>& v) const{
    if (v.empty()){
        return;
    }
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
}

//score candidates by squared L2 and keep top-N
std::vector<int> Hypercube::score_topN(const std::vector<double>& q, const std::vector<int>& cand, int N) const{
    if (cand.empty() || N <= 0){
        return {};
    }

    struct Pair{
        double dist; int id;
        bool operator<(const Pair& o) const { return dist < o.dist; }
    };

    std::vector<Pair> scored; scored.reserve(cand.size());
    for (int id : cand){
        scored.push_back({ l2_sq(q, (*data_ptr)[id]), id });
    }

    if ((int)scored.size() > N){
        std::nth_element(scored.begin(), scored.begin() + N, scored.end());
        scored.resize(N);
    } else{
        std::sort(scored.begin(), scored.end());
    }

    std::vector<int> result; result.reserve(scored.size());
    for (auto& p : scored){
        result.push_back(p.id);
    }

    return result;
}

//filter candidates with squared L2 <= R^2
std::vector<int> Hypercube::filter_range(const std::vector<double>& q, const std::vector<int>& cand, double R) const{
    std::vector<int> out;
    double R2 = R * R;
    for (int id : cand){
        if (l2_sq(q, (*data_ptr)[id]) <= R2){
            out.push_back(id);
        }
    }
    dedup_inplace(out);
    return out;
}

//approximate top-N via limited probes and checks
std::vector<int> Hypercube::ANN(const std::vector<double>& q, int N) const{
    if (!data_ptr || (int)q.size() != d || N <= 0) return {};
    
    if (!data_ptr){
        return {};
    }

    uint64_t start = vertex(q);

    std::vector<uint64_t> probe_ids;
    make_probes(start, probes, probe_ids);

    auto cand = collect_candidates(probe_ids);
    dedup_inplace(cand);

    return score_topN(q, cand, N);
}

//range search: same probes, keep ids within radius R
std::vector<int> Hypercube::Range(const std::vector<double>& q, double R) const{
    if (!data_ptr || (int)q.size() != d || R < 0.0) return {};
    
    if (!data_ptr){
        return {};
    }

    uint64_t start = vertex(q);
    std::vector<uint64_t> probe_ids;
    make_probes(start, probes, probe_ids);

    auto cand = collect_candidates(probe_ids);
    dedup_inplace(cand);

    return filter_range(q, cand, R);
}
