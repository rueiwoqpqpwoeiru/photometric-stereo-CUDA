
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <npp.h>
#include "Mem.h"
#include "Img_proc.h"
#include "Util.h"

template<typename T> class Captured_img {
public:
	Captured_img(const std::string& in_dir) {
		_img_list.read_csv(in_dir + "/use_img.txt");
		// get image size
		Img_io img0(in_dir + "/" + _img_list.get_str(0, 0));  // load the first image in the image list
		Tiff_tag tag = img0.tag;
		img0.close();
		// alloc memory
		_img.resize(tag.ch);
		for (unsigned int k = 0; k < _img.size(); k++) {
			_img[k].open_2d(tag.w, tag.h, _img_list.h());
		}
	}
	void imread(const std::string& in_dir, const std::string& out_dir) {
		// read img and store host memory
		for (unsigned int n = 0; n < _img_list.h(); n++) {
			Img_io img_io(in_dir + "/" + _img_list.get_str(0, n));
			Array<T> line_buf(img_io.tag.w * img_io.tag.ch);
			for (unsigned int i = 0; i < img_io.tag.h; i++) {
				img_io.read(line_buf.ptr, i);
				// RGBRGB... -> RR GG BB ..
				for (unsigned j = 0; j < img_io.tag.w; j++) {
					for (unsigned k = 0; k < img_io.tag.ch; k++) {
						_img[k].h_data.val(j, i, n) = line_buf.ptr[img_io.tag.ch * j + k];
					}
				}
			}
		}
		for (unsigned int k = 0; k < _img.size(); k++) {
			_img[k].h2d();
		}
		// for check
		_img[0].imwrite(out_dir + "/00_0_stacked_img_R.tif", PHOTOMETRIC_MINISBLACK);
		_img[1].imwrite(out_dir + "/00_1_stacked_img_G.tif", PHOTOMETRIC_MINISBLACK);
		_img[2].imwrite(out_dir + "/00_2_stacked_img_B.tif", PHOTOMETRIC_MINISBLACK);
	}
	const std::vector<D_mem<T> >& get() const { return _img; }
private:
	Data_frame _img_list;
	std::vector<D_mem<T> > _img;  //  RGB
};

// ch means number of light source (Due to processing reasons, the number of channels was fixed)
template<typename T, unsigned int ch> class Normal_map {
public:
	Normal_map(const std::string& in_dir) {
		// read light_mat
		_light_mat.read_csv(in_dir + "/lights.txt");
		// get img size
		Tiff_tag tag = _get_tifftag(in_dir);
		// alloc memory
		_normal_map.resize(3);  // RGB
		for (unsigned int k = 0; k < _normal_map.size(); k++) {
			_normal_map[k].open_2d(tag.w, tag.h, 4);    // 4ch means (nx, ny, nz, albedo)
		}
		_albedo_map.alloc(tag.w, tag.h);
	}
	void do_proc(const std::vector<D_mem<T> >& captured_img, const std::string& out_dir) {
		// calc normal
		for (unsigned int k = 0; k < _normal_map.size(); k++) {
			_calc_normal.do_proc(captured_img[k], _normal_map[k], _light_mat);  // k means RGB
		}
		_normal_map[0].imwrite(out_dir + "/01_0_normal_by_R.tif", PHOTOMETRIC_RGB);
		_normal_map[1].imwrite(out_dir + "/01_1_normal_by_G.tif", PHOTOMETRIC_RGB);
		_normal_map[2].imwrite(out_dir + "/01_2_normal_by_B.tif", PHOTOMETRIC_RGB);
		// make albedo
		_albedo_map.do_proc(_normal_map);
		_albedo_map.get_albedo_map().imwrite(out_dir + "/02_0_albedo_g10.tif", PHOTOMETRIC_RGB);
		_albedo_map.get_albedo_map_g22().imwrite(out_dir + "/02_1_albedo_g22.tif", PHOTOMETRIC_RGB);
	}
private:
	Tiff_tag _get_tifftag(const std::string& in_dir) {
		Data_frame img_list(in_dir + "/use_img.txt");
		Img_io img(in_dir + "/" + img_list.get_str(0, 0));  // load the first image in the image list
		Tiff_tag tag = img.tag;
		img.close();
		return tag;
	}
	Calc_normal<T, ch> _calc_normal;
	D_mem<float> _light_mat;
	std::vector<D_mem<T> > _normal_map;  // (nx, ny, nz, albedo) x (RGB)
	Albedo_map<T> _albedo_map;    // (albedo_R, albedo_G, albedo_B) x (g1.0, g2.2)
};

int main(int argc, char* argv[]) {
	const std::string in_dir = argv[1];
	const std::string out_dir = argv[2];
	// It takes time to allocate device memory, so allocate first
	Captured_img<uint16> captured_img(in_dir);
	Normal_map<uint16, 50> normal_map(in_dir);  // number of light source is fixed( if you want to change it, change the value "50" in the template )
	// please describe the loop for each in_dir and out_dir
	// now, to measure time, 5 times loop for same img
	for (unsigned int i = 0; i < 5; i++) {
		captured_img.imread(in_dir, out_dir);
		normal_map.do_proc(captured_img.get(), out_dir);
	}
    // end
	cudaDeviceReset();
	return 0;
};
