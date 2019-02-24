#define EDGE_PIXEL 70
#define EDGE_DISTANCE 8
#define EDGE_LENGTH 3
#define K_MAX 10

//CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math.h"

#include <stdio.h>

//Standard C++
#include <iostream>
#include <fstream>

#include <algorithm>
#include <string>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <cstddef>
#include <random>
#include <limits>
#include <iterator>
#include <tuple>
#include <functional>

//CImg
#include "CImg.h"
#define cimg_display 0

using namespace std;
using namespace cimg_library;

//Pixel Struct
typedef struct pixel{
	int x;
	int y;
	int r;
	int g;
	int b;
	bool is_edge = false;
	int kmean_group = -1;
}pixel;

//Image Struct
typedef struct image{
	vector<vector<pixel>> mat;
	int height;
	int width;
}image;

//Blob Struct
typedef struct blob{
	vector<pixel> blob_contents;
	pixel centroid;
	int area;
	int blob_num;
}blob;

//Feature Struct
typedef struct feature_info{
	string img_path;
	int x;
	int y;
	int height;
	int width;
}feature_info;


//CUDA Error Check
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		system("pause");
		if (abort) exit(code);
	}
}

//Standard Functions
image read_img(string file_name);

pixel* flatten_image(image img);

pixel* edge_detect(image img, pixel* h_array_img);

pair<vector<int>, vector<vector<pixel>>> kmeans(pixel* h_array_img, image img, int k_max);

pixel* elbow_method(vector<int>SSE, vector<vector<pixel>> array_k, int k_max);

pixel* crop_kmeans_img_carray(pixel *c_mat, int width, int height, int kmean_grp, int *crop_width, int *crop_height, image img,int iterator);

void gray_img_save(int width, int height, pixel* h_array_img);

void save_feature_info(feature_info feature_details);

//CUDA Kernels
__global__ void edge_detect(pixel *mat, int row_num, int col_num);

__global__ void km_assign_group(pixel *mat, pixel *means, int group_num, int row_num, int col_num);
