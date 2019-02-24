#include "globals.hpp"

pixel* flatten_image(image img){
	size_t img_bytes = (img.height * img.width) * sizeof(pixel);
	int num_elements = (img.height * img.width);

	pixel* h_array_img, *h_array_img_sv;
	h_array_img = (pixel*)malloc(img_bytes);
	h_array_img_sv = h_array_img;

	for (int i = 0; i < img.height; i++) {
		std::copy(img.mat[i].begin(), img.mat[i].end(), h_array_img);
		h_array_img += img.mat[i].size();
	}

	h_array_img = h_array_img_sv;
	return h_array_img;
}

pair<vector<int>, vector<vector<pixel>>> kmeans(pixel* h_array_img, image img,int k_max){
	pair<vector<int>, vector<vector<pixel>>> result;

	int blockSize, gridSize;

	// Number of threads in each thread block
	blockSize = 512;

	// Number of thread blocks in grid
	gridSize = (int)ceil((float)(img.height * img.width) / blockSize);

	size_t img_bytes = (img.height * img.width) * sizeof(pixel);
	int num_elements = (img.height * img.width);

	vector<pixel> means;
	//L: h_array_image has image data; pixel data in array

	//while (convergence == true){
	int num_groups;
	pixel *d_means;
	random_device random_device;
	mt19937 engine{ random_device() };
	uniform_int_distribution<int> dist(0, (img.height * img.width) - 1);

	//array_k has k rows. has h_array_image for the row where row matches k
	vector<vector<pixel>> array_k;
	pixel* pixel_ptr;
	vector<int> SSE;
	vector<float> slopeRatio;

	for (int km = 1; km <= k_max; km++) {
		num_groups = km;
		vector<pixel> k_centers;
		pixel c1;

		// find a random mean from an edge pixel

		for (int m = 0; m < km; m++) {
			while (c1.is_edge != true) {
				c1 = h_array_img[dist(engine)];
			}
			means.push_back(c1);
		}

		// code block is assigning k mean gropus
		pixel* d_array_img;

		gpuErrchk(cudaMalloc((void**)&d_array_img, img_bytes));
		gpuErrchk(cudaMemcpy(d_array_img, h_array_img, img_bytes, cudaMemcpyHostToDevice));

		gpuErrchk(cudaMalloc((void**)&d_means, sizeof(pixel) * num_groups));
		gpuErrchk(cudaMemcpy(d_means, &means[0], sizeof(pixel) * num_groups, cudaMemcpyHostToDevice));

		gpuErrchk(cudaMemcpy(d_array_img, h_array_img, img_bytes, cudaMemcpyHostToDevice));

		km_assign_group <<< gridSize, blockSize >>> (d_array_img, d_means, num_groups, img.height, img.width);

		gpuErrchk(cudaMemcpy(h_array_img, d_array_img, img_bytes, cudaMemcpyDeviceToHost));

		int iter = 0;
		vector<pixel> centroids;
		vector<pixel> prev_centroids;
		bool convergence = true;

		//look in this loop
		while (true) {


			pixel *d_centroids;

			// calculate current centroid based on x y coordinates of pixels ina group
			for (int i = 0; i < km; i++) {

				int x_sum = 0;
				int y_sum = 0;
				int xy_num = 0;
				pixel p_centroid;

				for (int j = 0; j < img.height * img.width; j++) {
					if (h_array_img[j].is_edge == true) {
						if (h_array_img[j].kmean_group == i) {
							x_sum += h_array_img[j].x;
							y_sum += h_array_img[j].y;
							xy_num++;
						}
					}
				}
				convergence = true;
				if (xy_num == 0) {
					centroids.push_back(means[i]);
					convergence = false;
					//system("pause");
				}
				else {
					p_centroid.kmean_group = i;
					p_centroid.x = (int)(x_sum / xy_num);
					p_centroid.y = (int)(y_sum / xy_num);
					centroids.push_back(p_centroid);
				}

			}


			gpuErrchk(cudaMalloc((void**)&d_centroids, sizeof(pixel) * num_groups));
			gpuErrchk(cudaMemcpy(d_centroids, &centroids[0], sizeof(pixel) * num_groups, cudaMemcpyHostToDevice));

			gpuErrchk(cudaMemcpy(d_array_img, h_array_img, img_bytes, cudaMemcpyHostToDevice));

			km_assign_group << < gridSize, blockSize >> > (d_array_img, d_centroids, num_groups, img.height, img.width);

			gpuErrchk(cudaMemcpy(h_array_img, d_array_img, img_bytes, cudaMemcpyDeviceToHost));


			//if (km == 1) {
			//goto done_kmeans;
			//}
			if (iter != 0) {

				double dist = 0.0;
				for (int k = 0; k < km; k++) {
					dist = abs(sqrt((centroids[k].x - prev_centroids[k].x) ^ 2 + (centroids[k].y - prev_centroids[k].y) ^ 2));
					float dist1 = abs((centroids[k].x - prev_centroids[k].x) ^ 2 + (centroids[k].y - prev_centroids[k].y) ^ 2);
					boolean flag = isnan(dist);

					//if (abs(sqrt((centroids[k].x - prev_centroids[k].x) ^ 2 + (centroids[k].y - prev_centroids[k].y) ^ 2)) > 0.000000000001 || isnan(dist) == true) {
					if (dist > 0.000000000001 || isnan(dist) == true){
						convergence = false;
					}
				}
				if (convergence == true) {
					goto done_kmeans;
				}
			}
			prev_centroids = centroids;
			centroids.clear();
			iter++;
		}//end of wile(true)
	done_kmeans:
		int g_sum = 0;
		for (int i = 0; i < km; i++) {
			int sum = 0;
			for (int j = 0; j < img.height * img.width; j++) {
				if (h_array_img[j].kmean_group == i) {
					//sum += (centroids[i].x - h_array_img[j].x) ^ 2;
					sum += (((centroids[i].x - h_array_img[j].x) * (centroids[i].x - h_array_img[j].x)) +
						((centroids[i].y - h_array_img[j].y) * (centroids[i].y - h_array_img[j].y)));
				}
			}
			g_sum += sum;
		}
		SSE.push_back(g_sum);
		vector<pixel> k_img;

		k_img.insert(k_img.begin(), h_array_img, h_array_img + (img.height * img.width));
		array_k.push_back(k_img);
	}

	result.first = SSE;
	result.second = array_k;

	return result;

}

pixel* elbow_method(vector<int>SSE, vector<vector<pixel>> array_k,int k_max){
	const int size_p = array_k[0].size();
	pixel* h_array_img;
	h_array_img = (pixel*)malloc(size_p * sizeof(pixel));

	vector<float> slopeRatio;
	int optimal_k = *max_element(SSE.rbegin(), SSE.rend()) + 1;
	//optimal_k = 1;

	for (int k = 0; k < k_max - 2; k++)
	{
		if (SSE[k + 1] - SSE[k + 2] != 0)
			slopeRatio.push_back(1.0 * (SSE[k] - SSE[k + 1]) / (SSE[k + 1] - SSE[k + 2]));
		else
			slopeRatio.push_back(0.0);
	}

	float max_slope = 0.0;

	for (int i = 0; i < k_max - 2; i++)
	{
		if (slopeRatio[i] > max_slope)
		{
			optimal_k = i + 2;
			max_slope = slopeRatio[i];
		}
	}
	vector<pixel> k_img = array_k[optimal_k - 1];
	memcpy(h_array_img,&k_img[0],k_img.size() * sizeof(pixel));

	return h_array_img;
}


__global__ void km_assign_group(pixel *mat, pixel *means, int group_num, int row_num, int col_num) {
	int tid = blockIdx.x*blockDim.x + threadIdx.x;

	if (tid >= (row_num * col_num)) {
		return;
	}
	int col_id = mat[tid].x;
	int row_id = mat[tid].y;

	if (col_id >= col_num || row_id >= row_num) {
		printf("tid out of bounds\n");
		return;
	}

	if (mat[tid].is_edge == true) {

		float dist = -1;
		int closest_group = -1;
		//printf("%d %d %d %d %d %d\n", means[0].x, means[0].y, means[1].x, means[1].y, means[2].x, means[2].y );
		for (int i = 0; i < group_num; i++) {
			float loc_dist = sqrtf(((col_id - means[i].x) * (col_id - means[i].x)) + ((row_id - means[i].y) * (row_id - means[i].y)));
			if (closest_group == -1) {
				closest_group = i;
				dist = loc_dist;
			}
			else if (loc_dist < dist) {
				closest_group = i;
				dist = loc_dist;
			}
		}
		mat[tid].kmean_group = closest_group;
		//printf("\n %d %d %f %d\n", col_id, row_id, dist, mat[tid].kmean_group);
	}
}
