#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <npp.h>
#include "Img_io.h"
#include "Util.h"

// host ptr ( page lock )
template<typename T> struct H_data {
	T* ptr;    // RRR GGG BBB ...
	unsigned int w;
	unsigned int h;
	unsigned int ch;
	void open(unsigned int w_, unsigned int h_, unsigned int ch_) {
		if (nullptr == ptr) {
			w = w_;	h = h_;	ch = ch_;
			ptr = static_cast<T*>(malloc(sizeof(T) * w * h * ch));
			cudaHostRegister(ptr, sizeof(T) * w * h * ch, cudaHostRegisterDefault);
			if (nullptr == ptr) { assert(false); };
		}
	}
	void close() {
		if (nullptr != ptr) {
			cudaHostUnregister(ptr);
			free(ptr);
			ptr = nullptr;
			w = 0; h = 0; ch = 0;
		}
	}
	H_data() {
		ptr = nullptr;
		w = 0; h = 0; ch = 0;
	}
	~H_data() {
		this->close();
	}
	// accessor
	T& val(unsigned int w_, unsigned int h_, unsigned int ch_) const { return *(ptr + w * (h * ch_ + h_) + w_); }
	// tif io (only 8bit or 16bit)
	void imread(const std::string& file_name) {
		if (2 < sizeof(T)) { assert(false); }
		Img_io img_io(file_name);
		Array<T> line_buf(w * ch);
		for (unsigned int i = 0; i < h; i++) {
			img_io.read(line_buf.ptr, i);
			// RGBRGB... -> RR GG BB ..
			for (unsigned j = 0; j < w; j++) {
				for (unsigned k = 0; k < ch; k++) {
					this->val(j, i, k) = line_buf.ptr[ch * j + k];
				}
			}
		}
	}
	void imwrite(const std::string& file_name, const int metric = PHOTOMETRIC_MINISBLACK) const {
		if (2 < sizeof(T)) { assert(false); }
		Tiff_tag tag;
		tag.set_val(w, h, ch, sizeof(T), metric);
		Img_io img_io(file_name, tag);
		Array<T> line_buf(w * ch);
		for (unsigned int i = 0; i < h; i++) {
			// RR GG BB .. -> RGBRGB...
			for (unsigned j = 0; j < w; j++) {
				for (unsigned k = 0; k < ch; k++) {
					line_buf.ptr[ch * j + k] = this->val(j, i, k);
				}
			}
			img_io.write(line_buf.ptr, i);
		}
	}
	// csv io (only	numeric)
	void read_csv(const std::string& file_name) {
		Data_frame df;
		df.read_csv(file_name);
		for (unsigned int i = 0; i < df.h(); i++) {
			for (unsigned int j = 0; j < df.w(); j++) {
				this->val(j, i, 0) = static_cast<float>(df.get_val(j, i));
			}
		}
	}
	void write_csv(const std::string& file_name) const {
		Data_frame df;
		df.resize(w, h * ch);
		for (unsigned k = 0; k < ch; k++) {
			for (unsigned int i = 0; i < h; i++) {
				for (unsigned int j = 0; j < w; j++) {
					df.set_val(j, h * k + i, static_cast<double>(this->val(j, i, k)));
				}
			}
		}
		df.write_csv(file_name);
	}
};

// device ptr
template<typename T> struct D_data {
	T* ptr;    // RRR GGG BBB ...
	cudaTextureObject_t tex;
	size_t pitch;
	unsigned int w;
	unsigned int h;
	unsigned int ch;
	__device__ __forceinline__ T& val(unsigned int w_, unsigned int h_, unsigned int ch_) const { return ((T*)((char*)(ptr) + pitch * (h * ch_ + h_)))[w_]; }
};

// device memory
template<typename T> struct D_mem {
	D_data<T> d_data;
	H_data<T> h_data;
	// open memory
	void open_2d(unsigned int w, unsigned int h, unsigned int ch) {
		if (nullptr == d_data.ptr) {
			d_data.w = w; d_data.h = h;	d_data.ch = ch;
			CUDA_CHECK(cudaMallocPitch(&(d_data.ptr), &(d_data.pitch), sizeof(T) * d_data.w, d_data.h * d_data.ch));
			h_data.open(d_data.w, d_data.h, d_data.ch);
		}
	}
	void open_1d(unsigned int w, unsigned int ch) {
		if (nullptr == d_data.ptr) {
			d_data.w = w; d_data.h = 1;	d_data.ch = ch;
			CUDA_CHECK(cudaMalloc(&(d_data.ptr), sizeof(T) * d_data.w * d_data.ch));
			d_data.pitch = 0;  // not use
			h_data.open(d_data.w, d_data.h, d_data.ch);
		}
	}
	void close() {
		if (nullptr != d_data.ptr) {
			cudaFree(d_data.ptr);
			d_data.ptr = nullptr;
			d_data.pitch = 0;
			d_data.w = 0; d_data.h = 0;	d_data.ch = 0;
			h_data.close();
		}
	}
	D_mem() {
		d_data.ptr = nullptr;
		d_data.pitch = 0;
		d_data.w = 0; d_data.h = 0;	d_data.ch = 0;
	}
	D_mem(unsigned int w, unsigned int h, unsigned int ch) {
		d_data.ptr = nullptr;
		d_data.pitch = 0;
		d_data.w = 0; d_data.h = 0;	d_data.ch = 0;
		this->open_2d(w, h, ch);
	}
	D_mem(unsigned int w, unsigned int h) {
		d_data.ptr = nullptr;
		d_data.pitch = 0;
		d_data.w = 0; d_data.h = 0;	d_data.ch = 0;
		this->open_1d(w, h);
	}
	~D_mem() {
		this->close();
	}
	// When using texture memory, call the following after open_2d
	void use_tex() {
		// Set boundary conditions, subpixel interpolation method, normalize coordinates, etc.
		cudaTextureDesc texDesc;
		memset(&texDesc, 0, sizeof(texDesc));
		texDesc.addressMode[0] = cudaAddressModeWrap;
		texDesc.addressMode[1] = cudaAddressModeWrap;
		texDesc.filterMode = cudaFilterModePoint;
		texDesc.normalizedCoords = false;
		texDesc.readMode = cudaReadModeElementType;
		// Set the attributes and parameters of the array to bind
		cudaResourceDesc resDesc;
		memset(&resDesc, 0, sizeof(resDesc));
		resDesc.res.pitch2D.desc = cudaCreateChannelDesc<T>();
		resDesc.res.pitch2D.pitchInBytes = d_data.pitch;
		resDesc.res.pitch2D.width = d_data.w;
		resDesc.res.pitch2D.height = d_data.h * d_data.ch;
		resDesc.res.pitch2D.devPtr = d_data.ptr;
		resDesc.resType = cudaResourceTypePitch2D;
		cudaCreateTextureObject(&(d_data.tex[i]), &resDesc, &texDesc, nullptr);
		CUDA_CHECK(cudaGetLastError());
	}
	void h2d() const {
		Stopwatch sw;
		if (1 < d_data.h) {
			CUDA_CHECK(cudaMemcpy2D(d_data.ptr, d_data.pitch, h_data.ptr, sizeof(T) * d_data.w, sizeof(T) * d_data.w, d_data.h * d_data.ch, cudaMemcpyHostToDevice));
		}
		else {
			CUDA_CHECK(cudaMemcpy(d_data.ptr, h_data.ptr, sizeof(T) * d_data.w * d_data.ch, cudaMemcpyHostToDevice));
		}
		sw.print_time(__FUNCTION__);
	}
	void d2h() const {
		Stopwatch sw;
		if (1 < d_data.h) {
			CUDA_CHECK(cudaMemcpy2D(h_data.ptr, sizeof(T) * d_data.w, d_data.ptr, d_data.pitch, sizeof(T) * d_data.w, d_data.h * d_data.ch, cudaMemcpyDeviceToHost));
		}
		else {
			CUDA_CHECK(cudaMemcpy(h_data.ptr, d_data.ptr, sizeof(T) * d_data.w * d_data.ch, cudaMemcpyDeviceToHost));
		}
		sw.print_time(__FUNCTION__);
	}
	// tif io (only 8bit or 16bit)
	void imread(const std::string& file_name) {
		// for get size
		Img_io img(file_name);
		// alloc memory if empty
		if (nullptr == d_data.ptr) {
			if (img.tag.bit == sizeof(T) * 8) {
				this->open_2d(img.tag.w, img.tag.h, img.tag.ch);
			}
			else {
				assert(false);
			}
		}
		// read img if size is OK
		if (img.tag.w == d_data.w && img.tag.h == d_data.h && img.tag.ch == d_data.ch && img.tag.bit == sizeof(T) * 8) {
			h_data.imread(file_name);
			this->h2d();
		}
		else {
			assert(false);
		}
	}
	void imwrite(const std::string& file_name, const int metric = PHOTOMETRIC_MINISBLACK) const {
		this->d2h();
		h_data.imwrite(file_name, metric);
	}
	// csv io (only	numeric)
	void read_csv(const std::string& file_name) {
		// for get size
		Data_frame df(file_name);
		// alloc memory if empty
		if (nullptr == d_data.ptr) {
			this->open_2d(df.w(), df.h(), 1);
		}
		// read img if size is OK
		if (df.w() == d_data.w && df.h() == d_data.h && 1 == d_data.ch) {
			h_data.read_csv(file_name);
			this->h2d();
		}
		else {
			assert(false);
		}
	}
	void write_csv(const std::string& file_name) const {
		this->d2h();
		h_data.write_csv(file_name);
	}
};
