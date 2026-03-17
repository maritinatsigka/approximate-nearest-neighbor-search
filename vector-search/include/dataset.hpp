#pragma once
#include <string>
#include <vector>

//supported dataset kinds
enum class DataType{
    MNIST,   //images: 28x28 bytes-> doubles in [0,1]
    SIFT   // float vectors from .sift/.fvecs-like binary
};

//loaders
bool load_mnist(const std::string& path, std::vector<std::vector<double>>& out);
bool load_sift(const std::string& path, std::vector<std::vector<double>>& out);

//dispatcher based on dataset type
inline bool load_dataset(const std::string& path, DataType type, std::vector<std::vector<double>>& out) {
    return (type == DataType::MNIST) ? load_mnist(path, out) : load_sift(path, out);
}