#include "dataset.hpp"
#include <fstream>
#include <cstdint>
#include <cstring>
#include <iostream>

//read a 32-bit big-endian unsigned int from stream
static uint32_t read_be32(std::ifstream& fin){ //read a 32-bit big-endian integer
    unsigned char b[4];
    fin.read(reinterpret_cast<char*>(b), 4);

    return (uint32_t(b[0]) << 24) | (uint32_t(b[1]) << 16) | (uint32_t(b[2]) <<  8) | uint32_t(b[3]);
}

//Load MNIST images. Output vectors are normalized to [0,1]
bool load_mnist(const std::string& path, std::vector<std::vector<double>>& out){
    std::ifstream fin(path, std::ios::binary);
    if (!fin){   //file open failed
        return false;
    }
    
    uint32_t magic = read_be32(fin); //expected 2051 for images
    if (magic != 2051){
        return false;   //wrong file type
    }

    uint32_t n = read_be32(fin);
    uint32_t rows = read_be32(fin);
    uint32_t cols = read_be32(fin);
    uint32_t dim = rows * cols;
    out.assign(n, std::vector<double>(dim));

    //read raw bytes
    for (uint32_t i = 0; i < n; ++i){
        for (uint32_t j = 0; j < dim; ++j){
            unsigned char v = 0;
            fin.read((char*)&v, 1);
            out[i][j] = v / 255.0;  //scale byte to [0,1]
        }
    }

    return true;
}

//Load SIFT-like binary: [int32 dim][dim floats] repeated until EOF.
//Floats are copied to doubles.
bool load_sift(const std::string& path, std::vector<std::vector<double>>& out){
    std::ifstream fin(path, std::ios::binary);
    if (!fin){  //file open failed
        return false;
    }

    out.clear();
    while (true){
        int32_t dimLE = 0;
        fin.read((char*)&dimLE, 4);  //read vector dim (little-endian host assumed)

        if (!fin){
            break;
        }

        int dim = dimLE;  //little-endian host assumed
        if (dim <= 0 || dim > 10000){
            return false;
        }

        std::vector<float> buf(dim);
        fin.read((char*)buf.data(), dim * sizeof(float));
        if (!fin){  //truncated file
            return false;
        }

        //copy/convert floats -> doubles
        std::vector<double> vec(dim);
        for (int i = 0; i < dim; ++i){
            vec[i] = (double)buf[i];
        }

        out.push_back(std::move(vec));
    }

    return !out.empty();  //at least one vector
}
