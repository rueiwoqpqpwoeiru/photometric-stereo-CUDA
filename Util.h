#pragma once
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <regex>
#include <iterator>

// instead of this func, use the macro below
inline void check_cudaError(const cudaError_t& err, const std::string& file, const int& line) {
	if (cudaSuccess != err) {
		std::cerr << cudaGetErrorString(err) << "\t" << file << "\t" << line << std::endl;
		exit(1);
	}
};

#define CUDA_CHECK(err) (check_cudaError(err, __FILE__, __LINE__))

class Stopwatch {
public:
	Stopwatch() {
		cudaEventCreate(&t1);
		cudaEventCreate(&t2);
		cudaEventRecord(t1);
	}
	void print_time(std::string func_name) {
		cudaEventRecord(t2);
		cudaEventSynchronize(t2);
		float m_sec = 0;
		cudaEventElapsedTime(&m_sec, t1, t2);
		std::cout << "time" << "\t" << func_name.c_str() << "\t" << m_sec << "\t(ms)" << std::endl;
	}
	~Stopwatch() {
		cudaEventDestroy(t1);
		cudaEventDestroy(t2);
	}
private:
	cudaEvent_t t1, t2;
};

template<typename T> struct Array {
	T* ptr;
	size_t size;
	void open(size_t size_) {
		if (nullptr == ptr) {
			size = size_;
			ptr = static_cast<T*>(malloc(sizeof(T) * size));
			if (nullptr == ptr) { assert(false); };
		}
	};
	void close() {
		if (nullptr != ptr) {
			free(ptr);
			ptr = nullptr;
			size = 0;
		}
	};
	void resize(size_t size) {
		this->close();
		this->open(size);
	};
	Array() {
		ptr = nullptr;
		size = 0;
	};
	Array(size_t size) {
		ptr = nullptr;
		this->open(size);
	};
	Array(const Array& array) {
		this->resize(array.size);
		for (size_t i = 0; i < size; i++) {
			ptr[i] = array.ptr[i];
		}
	};
	~Array() { this->close(); };
};

class Data_frame {
public:
	void read_csv(const std::string& file_name) {
		_d.clear();
		std::ifstream ifs(file_name);
		if (false == ifs.is_open()) {
			std::cerr << "failed to open " << file_name << std::endl;
			exit(-1);
		}
		std::string buf;
		size_t num_token = 0;  // for check
		while (getline(ifs, buf)) {
			std::regex pattern(",| |\t");  // delimiter
			std::vector<std::string> tokens;
			for (std::sregex_token_iterator i(std::begin(buf), std::end(buf), pattern, -1), end; i != end; i++) {
				tokens.push_back(*i);
			}
			if (0 == _d.size()) {
				num_token = tokens.size();
			}
			else if (tokens.size() != num_token) {
				assert(false);
			}
			_d.push_back(tokens);
		}
		std::cout << "read csv " << file_name << std::endl;
	}
	void write_csv(const std::string& file_name) const {
		std::ofstream ofs(file_name);
		if (false == ofs.is_open()) {
			std::cerr << "failed to open " << file_name << std::endl;
			exit(-1);
		}
		for (size_t i = 0; i < _d.size(); i++) {
			for (size_t j = 0; j < _d[i].size() - 1; j++) {
				ofs << _d[i][j] << ",";
			}
			ofs << _d[i][_d[i].size() - 1] << std::endl;
		}
		std::cout << "write csv " << file_name << std::endl;
	}
	void resize(size_t w, size_t h) {
		_d.clear();
		_d.resize(h);
		for (size_t i = 0; i < _d.size(); i++) {
			_d[i].resize(w);
		}
	}
	Data_frame(){}
	Data_frame(const std::string& file_name) {
		this->read_csv(file_name);
	}
	Data_frame(size_t w, size_t h) {
		this->resize(w, h);
	}
	size_t w() const { return _d[0].size(); }
	size_t h() const { return _d.size(); }
	std::string get_str(size_t w, size_t h) const { return _d[h][w]; }
	double      get_val(size_t w, size_t h) const { return std::stod(_d[h][w]); }
	void set_str(size_t w, size_t h, std::string s) { _d[h][w] = s; }
	void set_val(size_t w, size_t h, double v) { _d[h][w] = std::to_string(v); }
private:
	std::vector<std::vector<std::string> > _d;
};
