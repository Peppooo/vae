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

Eigen::VectorXf resize(Eigen::VectorXf& data,size_t w,size_t h,size_t ratio) {
	Eigen::VectorXf ret;
	ret.resize((w * h) / (ratio * ratio));
	size_t newW = w / ratio;
	size_t newH = h / ratio;
	std::cout << "resizing to " << newW << "x" << newH << std::endl;
	for(int j = 0; j < h; j += ratio) {
		for(int i = 0; i < w; i += ratio) {
			float mean = 0;
			for(int sj = 0; sj < ratio; sj++) {
				for(int si = 0; si < ratio; si++) {
					mean += data((i + si) + (j + sj) * w);
				}
			}
			ret((i + newW * j) / ratio) = mean / (ratio * ratio);
		}
	}
	return ret;
}

// nearest-neighbour resize and convert to grayscale normalized [0,1]

// Loads all PNG images found in the directory specified by `path`.
// If `path` is not a directory the function returns an empty vector.
// Each image is converted to a 784-length Eigen::VectorXf (28x28 grayscale, normalized [0,1]).
// Non-PNG files are skipped.
// this function was done first done with copilot then changed most of it cause it didnt understand un cazzo di niente
std::vector<Eigen::VectorXf> load_images(std::string path,std::vector<int>* labels,int max_files = -1) {
	if(labels) labels->clear();
	std::vector<Eigen::VectorXf> result;
	if(max_files > 1) result.reserve(max_files);
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
	int count = 0;
	if (h == INVALID_HANDLE_VALUE) {
		std::cerr << "load_images: cannot enumerate directory: " << path << "\n";
		return result;
	}
	do {
		// skip directories
		if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) continue;

		std::string name = fd.cFileName;

		std::string num = "";
		if(labels) {
			for(auto c : name) {
				if(c != '_') {
					num.push_back(c);
				}
				else {
					break;
				}
			}
		}

		std::string full = path + name;

		int w = 0, hgt = 0, channels = 0;
		unsigned char* data = stbi_load(full.c_str(), &w, &hgt, &channels, 3);
		if (!data) {
			std::cerr << "load_images: failed to load: " << full << " (" << stbi_failure_reason() << ")\n";
			continue;
		}
		if(labels) {
			labels->push_back(std::stoi(num));
		}
		Eigen::VectorXf vec;
		vec.resize(w * hgt);
		for(int j = 0; j < hgt; j++) {
			for(int i = 0; i < w; i++) {
				uint8_t* c = data + (i + j * w)*3;
				if(c[0] == c[1] && c[1] == c[2] && c[2] == c[0]) {
					vec[i + j * w] = c[0]/255.0;
				}
				else {
					vec[i + j * w] = (0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]) / 255;
				}
			}
		}
		count++;
		if(count >= max_files) return result;
		if(count % 200 == 0) {
			std::cout << "load_images count: " << count << std::endl;
		}
		
		stbi_image_free(data);
		result.push_back(std::move(vec));
		//result.push_back(std::move(resize(vec,70,80,2)));
	} while (FindNextFileA(h, &fd));
	FindClose(h);

	std::cout << "loaded " << result.size() << " images from " << path << std::endl;

	return result;
}

std::vector<Eigen::VectorXf> load_images_rgb(std::string path,std::vector<int>* labels,int max_files = -1) {
	if(labels) labels->clear();
	std::vector<Eigen::VectorXf> result;
	if(max_files != -1) {
		result.reserve(max_files);
	}
	else {
		max_files = INT_MAX;
	}
	if(path.empty()) return result;

	DWORD attrs = GetFileAttributesA(path.c_str());
	if(attrs == INVALID_FILE_ATTRIBUTES || !(attrs & FILE_ATTRIBUTE_DIRECTORY)) {
		std::cerr << "load_images: path is not a directory or cannot be accessed: " << path << "\n";
		return result; // user requested path must not be a single image
	}

	// Ensure trailing backslash for concatenation
	if(path.back() != '\\' && path.back() != '/') path += "\\";

	std::string pattern = path + "*";
	WIN32_FIND_DATAA fd;
	HANDLE h = FindFirstFileA(pattern.c_str(),&fd);
	int count = 0;
	if(h == INVALID_HANDLE_VALUE) {
		std::cerr << "load_images: cannot enumerate directory: " << path << "\n";
		return result;
	}
	do {
		// skip directories
		if(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) continue;

		std::string name = fd.cFileName;

		std::string num = "";
		if(labels) {
			for(auto c : name) {
				if(c != '_') {
					num.push_back(c);
				}
				else {
					break;
				}
			}
		}

		std::string full = path + name;

		int w = 0,hgt = 0,channels = 0;
		unsigned char* data = stbi_load(full.c_str(),&w,&hgt,&channels,3);
		if(!data) {
			std::cerr << "load_images: failed to load: " << full << " (" << stbi_failure_reason() << ")\n";
			continue;
		}
		if(labels) {
			labels->push_back(std::stoi(num));
		}
		Eigen::VectorXf vec;
		vec.resize(w * hgt * 3);
		for(int j = 0; j < hgt; j++) {
			for(int i = 0; i < w; i++) {
				uint8_t* c = data + (i + j * w) * 3;
				vec[(i + j * w) * 3] = c[0] / 255.0;
				vec[(i + j * w) * 3+1] = c[1] / 255.0;
				vec[(i + j * w) * 3+2] = c[2] / 255.0;
			}
		}
		count++;
		if(count >= max_files) return result;
		if(count % 200 == 0) {
			std::cout << "load_images count: " << count << std::endl;
		}

		stbi_image_free(data);
		result.push_back(std::move(vec));
		//result.push_back(std::move(resize(vec,70,80,2)));
	} while(FindNextFileA(h,&fd));
	FindClose(h);

	std::cout << "loaded " << result.size() << " images from " << path << std::endl;

	return result;
}