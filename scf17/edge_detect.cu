#include "globals.hpp"

pixel* edge_detect(image img, pixel* h_array_img){

	size_t img_bytes = (img.height * img.width) * sizeof(pixel);
	int num_elements = (img.height * img.width);

	pixel* d_array_img;

	gpuErrchk(cudaMalloc((void**)&d_array_img, img_bytes));
	gpuErrchk(cudaMemcpy(d_array_img, h_array_img, img_bytes, cudaMemcpyHostToDevice));

	int blockSize, gridSize;

	// Number of threads in each thread block
	blockSize = 512;

	// Number of thread blocks in grid
	gridSize = (int)ceil((float)(img.height * img.width) / blockSize);

	edge_detect << < gridSize, blockSize >> >(d_array_img, img.height, img.width);

	gpuErrchk(cudaMemcpy(h_array_img, d_array_img, img_bytes, cudaMemcpyDeviceToHost));

	return h_array_img;
}

__global__ void edge_detect(pixel *mat, int row_num, int col_num) {

	int tid = blockIdx.x*blockDim.x + threadIdx.x;

	if (tid >= (row_num * col_num)) {
		return;
	}
	int x_val = mat[tid].x;
	int y_val = mat[tid].y;
	int g_val = mat[tid].g;
	bool edge_val = mat[tid].is_edge;

	int col_id = x_val;
	int row_id = y_val;

	if (col_id >= col_num || row_id >= row_num) {
		printf("tid out of bounds/n");
		return;
	}

	int p = mat[row_id * col_num + col_id].g;
	//horizontal

	for (int i = col_id; i < col_id + EDGE_LENGTH; i++) {

		if (i >= col_num) {
			break;
		}
		int g = mat[row_id * col_num + i].g;
		if (abs(p - g) > EDGE_PIXEL) {
			mat[row_id * col_num + col_id].is_edge = true;
		}

	}

	for (int i = col_id; i > col_id - EDGE_LENGTH; i--) {
		if (i < 0) {
			break;
		}

		int g = mat[row_id * col_num + i].g;
		if (abs(p - g) > EDGE_PIXEL) {
			mat[row_id * col_num + col_id].is_edge = true;
		}

	}

	//vertical
	for (int i = row_id; i < row_id + EDGE_LENGTH; i++) {
		if (i >= row_num) {
			break;
		}

		int g = mat[i * col_num + col_id].g;
		if (abs(p - g) > EDGE_PIXEL) {
			mat[row_id * col_num + col_id].is_edge = true;
		}

	}

	for (int i = row_id; i > row_id - EDGE_LENGTH; i--) {
		if (i < 0) {
			break;
		}

		int g = mat[i * col_num + col_id].g;
		if (abs(p - g) > EDGE_PIXEL) {
			mat[row_id * col_num + col_id].is_edge = true;
		}
	}
}