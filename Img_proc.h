#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <npp.h>
#include "Mem.h"
#include "Util.h"

struct Proc_unit {
	dim3 block;
	dim3 grid;
	Proc_unit(unsigned int w, unsigned int h, dim3 block_) {
		set(w, h, block_);
	}
	void set(unsigned int w, unsigned int h, dim3 block_) {
		block = block_;
		grid = dim3((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
	};
};

// kernel func
namespace K_func {
	// call a fixed-name method that named "kernel" in each class
	template<typename T, class... Args> __global__ void generic_kernel(T* obj, Args... args) {
		(obj->kernel)(args...);
	}
	__device__ __forceinline__ uint16 clip_uint16(const float& val) {
		uint16 ret = 0;
		if (0.0 > val) {
			ret = 0;
		}
		else if (65535.0 < val) {
			ret = 65535;
		}
		else {
			ret = static_cast<uint16>(val + 0.5);
		}
		return ret;
	}
	// range(-1, 1) ---> range(0, 65535)
	__device__ __forceinline__ uint16 trans_normal_to_uint16img(const float& val) {
		return K_func::clip_uint16((val + 1.0) / 2.0 * 65535.0);
	}
};

// static array for cuda kernel by using c++ template (warning: malloc waste 10ms order)
template<typename T, unsigned int w, unsigned int h> struct K_mat {
	T val[h][w];
	__device__ void transpose(const K_mat<T, h, w>& in) {
		for (unsigned int i = 0; i < h; i++) {
			for (unsigned int j = 0; j < w; j++) {
				val[i][j] = in.val[j][i];
			}
		}
	}
	template<unsigned int d> __device__ void mul(const K_mat<T, d, h>& in0, const K_mat<T, w, d>& in1) {
		for (unsigned int i = 0; i < h; i++) {
			for (unsigned int j = 0; j < w; j++) {
				val[i][j] = 0.0;
				for (unsigned int k = 0; k < d; k++) {
					val[i][j] = val[i][j] + in0.val[i][k] * in1.val[k][j];
				}
			}
		}
	}
	__device__ void inverse_3x3(const K_mat<T, h, w>& in) {
		float det = abs(in.val[0][0] * in.val[1][1] * in.val[2][2]
			          + in.val[0][1] * in.val[1][2] * in.val[2][0]
			          + in.val[0][2] * in.val[2][1] * in.val[1][0]
                                  - in.val[0][2] * in.val[1][1] * in.val[2][0]
			          - in.val[0][0] * in.val[2][1] * in.val[1][2]
			          - in.val[0][1] * in.val[1][0] * in.val[2][2]);
		if (1e-7 < det) {  // machine epsilon of 32bit is 1.192e-7
			for (unsigned int i = 0; i < h; i++) {
				for (unsigned int j = 0; j < w; j++) {
					val[i][j] = in.val[(i + 1) % 3][(j + 1) % 3] * in.val[(i + 2) % 3][(j + 2) % 3]
						  - in.val[(i + 2) % 3][(j + 1) % 3] * in.val[(i + 1) % 3][(j + 2) % 3];
					val[i][j] = __fdividef(val[i][j], det);
				}
			}
		}
		else {
			for (unsigned int i = 0; i < h; i++) {
				for (unsigned int j = 0; j < w; j++) {
					val[i][j] = 0.0;
				}
			}
		}
	}
	__device__ float normalize(const K_mat<T, 1, h>& in) {
		float sum = 0.0;
		for (unsigned int i = 0; i < h; i++) {
			sum = sum + in.val[i][0] * in.val[i][0];
		}
		float norm = sqrt(sum);
		if (1e-7 < norm) {
			for (unsigned int i = 0; i < h; i++) {
				val[i][0] = __fdividef(in.val[i][0], norm);
			}
		}
		else {
			for (unsigned int i = 0; i < h; i++) {
				val[i][0] = 0.0;
			}
		}
		return norm;
	}
};

// ch means number of light source
template<typename T, unsigned int ch> class Calc_normal {
public:
	void do_proc(const D_mem<T>& in, D_mem<T>& out, const D_mem<float>& light_mat, const dim3 block = dim3(32, 8)) {
		Proc_unit p(in.d_data.w, in.d_data.h, block);
		Stopwatch sw;
		K_func::generic_kernel<Calc_normal> << < p.grid, p.block >> > (this, in.d_data, out.d_data, light_mat.d_data);
		sw.print_time(__FUNCTION__);
	}
	__device__ __forceinline__ void calc_pseudo_normal_vector(const K_mat<float, 1, ch>& mat_I, const K_mat<float, 3, ch>& mat_L, K_mat<float, 1, 3>& mat_pN) {
		// L^T
		K_mat<float, ch, 3> mat_L_T;
		mat_L_T.transpose(mat_L);
		// L^T * L
		K_mat<float, 3, 3> mat_LTL;
		mat_LTL.mul(mat_L_T, mat_L);
		// (L^T * L)^-1
		K_mat<float, 3, 3> mat_LTL_inv;
		mat_LTL_inv.inverse_3x3(mat_LTL);
		// L^T * I
		K_mat<float, 1, 3> mat_LTI;
		mat_LTI.mul(mat_L_T, mat_I);
		// (L^T * L)^-1 * L^T * I   (pseudo normal vector)
		mat_pN.mul(mat_LTL_inv, mat_LTI);
	};
	__device__ __forceinline__ void kernel(const D_data<T> in, D_data<T> out, D_data<float> light_mat) {
		unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
		unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
		if (in.w > x && in.h > y) {
			// luminance vec ("I" means intensity)
			K_mat<float, 1, ch> mat_I;
			// mat of light vec
			K_mat<float, 3, ch> mat_L;
			// get val
			for (unsigned int i = 0; i < ch; i++) {
				mat_I.val[i][0] = static_cast<float>(in.val(x, y, i));
				for (unsigned int j = 0; j < 3; j++) {
					mat_L.val[i][j] = static_cast<float>(light_mat.val(j, i, 0));
				}
			}
			// pseudo normal vector
			K_mat<float, 1, 3> mat_pN;
			this->calc_pseudo_normal_vector(mat_I, mat_L, mat_pN);
			// normal vector and albedo
			K_mat<float, 1, 3> mat_N;
			float albedo = mat_N.normalize(mat_pN);
			// set val
			for (unsigned int k = 0; k < 3; k++) {
				out.val(x, y, k) = K_func::trans_normal_to_uint16img(mat_N.val[k][0]);
			}
			out.val(x, y, 3) = K_func::clip_uint16(albedo);
		}
	}
};

template<typename T> class Albedo_map {
public:
	void alloc(unsigned int w, unsigned int h) {
		_albedo_map.open_2d(w, h, 3);  // RGB
		_albedo_map_g22.open_2d(w, h, 3);
	}
	void do_proc(const std::vector<D_mem<T> >& in, const dim3 block = dim3(32, 8)) {
		Proc_unit p(in[0].d_data.w, in[0].d_data.h, block);
		Stopwatch sw;
		K_func::generic_kernel<Albedo_map> << < p.grid, p.block >> > (this, in[0].d_data, in[1].d_data, in[2].d_data, _albedo_map.d_data, _albedo_map_g22.d_data);
		sw.print_time(__FUNCTION__);
	}
	__device__ __forceinline__ void kernel(const D_data<T> in0, const D_data<T> in1, const D_data<T> in2, D_data<T> out0, D_data<T> out1) {
		unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
		unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
		if (in0.w > x && in0.h > y) {
			for (unsigned int k = 0; k < 3; k++) {
				// _albedo_map
				out0.val(x, y, 0) = in0.val(x, y, 3);   // ch 0 to 2 is xyz, 3 is albedo
				out0.val(x, y, 1) = in1.val(x, y, 3);
				out0.val(x, y, 2) = in2.val(x, y, 3);
				// _albedo_map_g22
				out1.val(x, y, 0) = K_func::clip_uint16(__powf(__fdividef(out0.val(x, y, 0), 65535.0), 0.4545) * 65535.0);
				out1.val(x, y, 1) = K_func::clip_uint16(__powf(__fdividef(out0.val(x, y, 1), 65535.0), 0.4545) * 65535.0);
				out1.val(x, y, 2) = K_func::clip_uint16(__powf(__fdividef(out0.val(x, y, 2), 65535.0), 0.4545) * 65535.0);
			}
		}
	}
	const D_mem<T>& get_albedo_map() const { return _albedo_map; }
	const D_mem<T>& get_albedo_map_g22() const { return _albedo_map_g22; }
private:
	D_mem<T> _albedo_map;      // gamma 1.0 (Equivalent to reflectance)
	D_mem<T> _albedo_map_g22;  // gamma 2.2 (Equivalent to lightness)
};
