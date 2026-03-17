#include "utils.hpp"
#include <cmath>
#include <algorithm>
#include <cstddef>

//Euclidean L2 distance
double l2(const std::vector<double>& a, const std::vector<double>& b){
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i){
        double d = a[i] - b[i];
        sum += d * d;
    }
    return std::sqrt(sum);
}

//Squared L2 distance, avoids sqrt for speed
double l2_sq(const std::vector<double>& a, const std::vector<double>& b){
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i){
        double d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

//brute-force Top-N by distance
std::vector<int> brute_force_topN(const std::vector<std::vector<double>>& data, const std::vector<double>& q, int N){
    if (N <= 0){
        return {};
    }
    
    struct Pair{
        double dist;
        int id;
        bool operator<(const Pair& other) const { return dist < other.dist; }
    };

    std::vector<Pair> vals;
    vals.reserve(data.size());
    for (int i = 0; i < (int)data.size(); ++i){
        vals.push_back({ l2(q, data[i]), i });
    }

    if ((int)vals.size() > N){
        std::nth_element(vals.begin(), vals.begin() + N, vals.end());
        vals.resize(N);
    }

    std::vector<int> ids;
    ids.reserve(vals.size());
    for (auto& p : vals) ids.push_back(p.id);
    return ids;
}

//brute-force range: collect ids with L2 <= R
std::vector<int> brute_force_range(const std::vector<std::vector<double>>& data, const std::vector<double>& q, double R){
    std::vector<int> ids;
    for (int i = 0; i < (int)data.size(); ++i){
        if (l2(q, data[i]) <= R){
            ids.push_back(i);
        }
    }
    return ids;
}

//exact range using squared distances: compare with R^2 
std::vector<int> range_search(const std::vector<std::vector<double>>& data, const std::vector<double>& q,double R){
    std::vector<int> result;
    double R2 = R * R;

    for (int id = 0; id < (int)data.size(); ++id){
        if (l2_sq(q, data[id]) <= R2){
            result.push_back(id);
        }
    }

    return result;
}

//validate all vectors share the same dimensionality
bool check_consistent_dim(const std::vector<std::vector<double>>& X){
    if (X.empty()){
        return true;
    }
    const std::size_t D = X[0].size();
    for (const auto& v : X){
        if (v.size() != D){
            return false;
        }
    }
    return true;
}