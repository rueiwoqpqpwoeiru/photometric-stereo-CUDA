
#pragma once

#include <cassert>
#include <string>
#include <iostream>
#include "./tiffio.h"

struct Tiff_tag{
    int metric;     // photometric
    int ch;
    int bit;
    int w;          // width
    int h;          // height
    int rows;       // height
    int dpiUnit;
    float dpiW;
    float dpiH;
    int compress;
    int plnrCfg;    // planarconfig
    int orient;     // orientation
    Tiff_tag(){
        metric    = 1;
        ch        = 1;
        bit       = 8;
        w         = 1;
        h         = 1;
        rows      = 1;
        dpiUnit   = 2;
        dpiW      = 72.0;
        dpiH      = 72.0;
        compress  = 1;
        plnrCfg   = 1;
        orient    = 1;
    };
    void get(TIFF* tifPtr){
        TIFFGetField( tifPtr, TIFFTAG_PHOTOMETRIC,        &metric );
        TIFFGetField( tifPtr, TIFFTAG_SAMPLESPERPIXEL,    &ch );
        TIFFGetField( tifPtr, TIFFTAG_BITSPERSAMPLE,      &bit );
        TIFFGetField( tifPtr, TIFFTAG_IMAGEWIDTH,         &w );
        TIFFGetField( tifPtr, TIFFTAG_IMAGELENGTH,        &h );
        TIFFGetField( tifPtr, TIFFTAG_ROWSPERSTRIP,       &rows );
        TIFFGetField( tifPtr, TIFFTAG_RESOLUTIONUNIT,     &dpiUnit );
        TIFFGetField( tifPtr, TIFFTAG_XRESOLUTION,        &dpiW );
        TIFFGetField( tifPtr, TIFFTAG_YRESOLUTION,        &dpiH );
        TIFFGetField( tifPtr, TIFFTAG_COMPRESSION,        &compress );
        TIFFGetField( tifPtr, TIFFTAG_PLANARCONFIG,       &plnrCfg );
        TIFFGetField( tifPtr, TIFFTAG_ORIENTATION,        &orient );
    };
    void set(TIFF* tifPtr){
        TIFFSetField( tifPtr, TIFFTAG_PHOTOMETRIC,        metric );
        TIFFSetField( tifPtr, TIFFTAG_SAMPLESPERPIXEL,    ch );
        TIFFSetField( tifPtr, TIFFTAG_BITSPERSAMPLE,      bit );
        TIFFSetField( tifPtr, TIFFTAG_IMAGEWIDTH,         w );
        TIFFSetField( tifPtr, TIFFTAG_IMAGELENGTH,        h );
        TIFFSetField( tifPtr, TIFFTAG_ROWSPERSTRIP,       rows );
        TIFFSetField( tifPtr, TIFFTAG_RESOLUTIONUNIT,     dpiUnit );
        TIFFSetField( tifPtr, TIFFTAG_XRESOLUTION,        dpiW );
        TIFFSetField( tifPtr, TIFFTAG_YRESOLUTION,        dpiH );
        TIFFSetField( tifPtr, TIFFTAG_COMPRESSION,        compress );
        TIFFSetField( tifPtr, TIFFTAG_PLANARCONFIG,       plnrCfg );
        TIFFSetField( tifPtr, TIFFTAG_ORIENTATION,        orient );
    };
    void set_val(unsigned int w_, unsigned int h_, unsigned int ch_, unsigned int byte, int metric_ = PHOTOMETRIC_MINISBLACK) {
        w = w_;
        h = h_;
        ch = ch_;
        bit = 8 * byte;
        metric = metric_;
    }
};

class Img_io{
public:
    // read mode
    void open( std::string file_name ){
        if(nullptr == _ptr){
            _ptr = TIFFOpen( file_name.c_str(), "r" );
            if(nullptr == _ptr ){
                std::cerr << "failed to open " << file_name << std::endl;
                exit(-1);
            }
            tag.get( _ptr );
            std::cout << "open tif (read mode) " << file_name << std::endl;
        }
    };
    // write mode
    void open( std::string file_name, Tiff_tag tag_in ){
        if(nullptr == _ptr){
            _ptr = TIFFOpen( file_name.c_str(), "w" );
            if(nullptr == _ptr ){
                std::cerr << "failed to open " << file_name << std::endl;
                exit(-1);
            }
            tag = tag_in;
            tag.set( _ptr );
            std::cout << "open tif (write mode) " << file_name << std::endl;
        }
    };
    void close(){
        if(_ptr != nullptr){
            TIFFClose( _ptr );
            _ptr = nullptr;
        }
    };
    // read mode
    Img_io( std::string file_name ){
        _ptr = nullptr;
        this->open( file_name );
    };
    // write mode
    Img_io( std::string file_name, Tiff_tag tag_in ){
        _ptr = nullptr;
        this->open( file_name, tag_in );
    };
    ~Img_io(){ this->close(); };
    // read write
    void read( tdata_t line_buf, uint32 row)  const { TIFFReadScanline(  _ptr, line_buf, row, 0); };
    void write(tdata_t line_buf, uint32 row ) const { TIFFWriteScanline( _ptr, line_buf, row, 0); };
    // Tag is changed directly
    Tiff_tag tag;
private:
    TIFF* _ptr;
};
