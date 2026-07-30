#pragma once
#include <iostream>
#include <string>
#include <algorithm>
#include <vector>
#include <windows.h>
#include <Eigen/Eigen>

// stb_image single-header used to decode PNGs. Ensure stb_image.h is available in the project include path.
#ifndef STB_IMAGE_IMPLEMENTATION_LOADER
#define STB_IMAGE_IMPLEMENTATION_LOADER
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#endif

static std::string to_lower(const std::string &s) {
	std::string out = s;
	std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) { return std::tolower(c); });
	return out;
}

// nearest-neighbour resize and convert to grayscale normalized [0,1]

// Loads all PNG images found in the directory specified by `path`.
// If `path` is not a directory the function returns an empty vector.
// Each image is converted to a 784-length Eigen::VectorXd (28x28 grayscale, normalized [0,1]).
// Non-PNG files are skipped.
std::vector<Eigen::VectorXd> load_images(std::string path) {
	std::vector<Eigen::VectorXd> result;
	if (path.empty()) return result;

	DWORD attrs = GetFileAttributesA(path.c_str());
	if (attrs == INVALID_FILE_ATTRIBUTES || !(attrs & FILE_ATTRIBUTE_DIRECTORY)) {
		std::cerr << "load_images: path is not a directory or cannot be accessed: " << path << "\n";
		return result; // user requested path must not be a single image
	}

	// Ensure trailing backslash for concatenation
	if (path.back() != '\\' && path.back() != '/') path += "\\";

	std::string pattern = path + "*";
	WIN32_FIND_DATAA fd;
	HANDLE h = FindFirstFileA(pattern.c_str(), &fd);
	if (h == INVALID_HANDLE_VALUE) {
		std::cerr << "load_images: cannot enumerate directory: " << path << "\n";
		return result;
	}
	int asd = 0;
	do {
		// skip directories
		if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) continue;

		std::string name = fd.cFileName;

		std::string full = path + name;

		int w = 0, hgt = 0, channels = 0;
		unsigned char* data = stbi_load(full.c_str(), &w, &hgt, &channels, 3);
		if (!data) {
			std::cerr << "load_images: failed to load: " << full << " (" << stbi_failure_reason() << ")\n";
			continue;
		}
		if(asd <= 25) {
			std::cout << name << std::endl;

		}
		asd++;
		Eigen::VectorXd vec;
		vec.resize(w * hgt);
		for(int i = 0; i < w; i++) {
			for(int j = 0; j < hgt; j++) {
				uint8_t* c = data + (i + j * w)*3;
				vec[i+j*w] = (0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2])/255;
			}
		}

		stbi_image_free(data);
		result.push_back(std::move(vec));
	} while (FindNextFileA(h, &fd));
	FindClose(h);

	std::cout << "loaded " << result.size() << " images from " << path << std::endl;

	return result;
}