#pragma once
#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <Eigen/Eigen>

namespace mnist {

    uint32_t swap32(uint32_t v) {
        return (v >> 24) |
            ((v >> 8) & 0x0000FF00) |
            ((v << 8) & 0x00FF0000) |
            (v << 24);
    }

    uint32_t readBigEndian(std::ifstream& file) {
        uint32_t result = 0;
        file.read(reinterpret_cast<char*>(&result),sizeof(result));
        return swap32(result);
    }

    void read_images(const char* filename,std::vector<Eigen::Vector<double,784>>& Y) {
        std::ifstream file(filename,std::ios::binary);
        if(!file) {
            std::cerr << "Cannot open image file\n";
        }

        std::uint32_t magic = readBigEndian(file);
        std::uint32_t numImages = readBigEndian(file);
        std::uint32_t rows = readBigEndian(file);
        std::uint32_t cols = readBigEndian(file);

        std::vector<std::uint8_t> images(numImages * rows * cols);
        file.read(reinterpret_cast<char*>(images.data()),images.size());

        //cout << "Magic: " << magic << "\n";
        std::cout << "Images: " << numImages << "\n";
        std::cout << "Size: " << rows << "x" << cols << "\n";


        Y.clear();

        Y.resize(numImages);
        for(int i = 0; i < numImages; i++) {
            for(int j = 0; j < (28 * 28); j++) {
                Y[i](j) =
                    (images[i * 28ull * 28 + j] / 255.0);
            }
        }
    }

    void read_labels(const char* filename,std::vector<Eigen::Vector<double,10>>* X,std::vector<uint8_t>& labels) {
        std::ifstream file(filename,std::ios::binary);

        std::uint32_t magic = readBigEndian(file);
        std::uint32_t numLabels = readBigEndian(file);

        labels = std::vector<uint8_t>(numLabels);
        file.read(reinterpret_cast<char*>(labels.data()),labels.size());

        if(X) (*X).resize(numLabels);

        for(int i = 0; i < numLabels; i++) {
            std::uint8_t label = labels[i];
            if(X) {
                for(int j = 0; j < 10; j++) {
                    (*X)[i][j] = (j == label);
                }
            }
        }
    }

}