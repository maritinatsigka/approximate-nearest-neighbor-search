#pragma once
#include <vector>

//L2 distance (sqrt) and squared L2 (no sqrt)
double l2(const std::vector<double>& a, const std::vector<double>& b);
double l2_sq(const std::vector<double>& a, const std::vector<double>& b);

//brute force helpers
//top-N nearest ids
std::vector<int> brute_force_topN(const std::vector<std::vector<double>>& data, const std::vector<double>& q, int N);

//all ids within radius R (uses true L2 with sqrt)
std::vector<int> brute_force_range(const std::vector<std::vector<double>>& data, const std::vector<double>& q, double R);

//exact range search using squared L2 (R^2 comparison)
std::vector<int> range_search(const std::vector<std::vector<double>>& data, const std::vector<double>& q, double R);

//checks that all vectors in X have the same dimensionality.
bool check_consistent_dim(const std::vector<std::vector<double>>& X);