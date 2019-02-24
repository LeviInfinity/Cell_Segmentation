#include "globals.hpp"

image read_img(string file_name){
	vector<image> train_data_color;
	vector<image> train_data_grayscale;

	int height;
	int width;
	vector<string> train_files;
	train_files.push_back(file_name);

	//Getting RGB values of image
	for (auto file : train_files) {
		image data;
		image datag;
		CImg<unsigned char> src(file.c_str());

		width = src.width();
		height = src.height();

		data.mat.resize(height);
		for (int i = 0; i < height; i++)
			data.mat[i].resize(width);

		datag.mat.resize(height);
		for (int i = 0; i < height; i++)
			datag.mat[i].resize(width);

		for (int r = 0; r < height; r++) {
			for (int c = 0; c < width; c++) {
				pixel p;
				p.x = c;
				p.y = r;
				p.r = (int)src(c, r, 0, 0);
				p.g = (int)src(c, r, 0, 1);
				p.b = (int)src(c, r, 0, 2);
				pixel g;

				//Grayscale
				int x = (0.299 * p.r) + (0.587 * p.g) + (0.114 * p.b);
				g.x = c;
				g.y = r;
				g.r = x;
				g.g = x;
				g.b = x;
				data.mat[r][c] = p;
				data.height = height;
				data.width = width;
				datag.mat[r][c] = g;
				datag.height = height;
				datag.width = width;
			}
		}
		train_data_color.push_back(data);
		train_data_grayscale.push_back(datag);
	}
	cout << endl;
	return train_data_grayscale[0];
}



pixel* crop_kmeans_img_carray(pixel *c_mat, int width, int height, int kmean_grp, int *crop_width, int *crop_height,image img,int iterator) {

	int min_x = width - 1;
	int max_x = 0;
	int min_y = height - 1;
	int max_y = 0;

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (c_mat[i * width + j].kmean_group == kmean_grp) {
				if (c_mat[i * width + j].x < min_x)
					min_x = c_mat[i * width + j].x;
				if (c_mat[i * width + j].y < min_y)
					min_y = c_mat[i * width + j].y;

				if (c_mat[i * width + j].x > max_x)
					max_x = c_mat[i * width + j].x;
				if (c_mat[i * width + j].y > max_y)
					max_y = c_mat[i * width + j].y;
			}
		}
	}


	int new_width = max_x - min_x + 1;
	int new_height = max_y - min_y + 1;

	int x_val = ( (min_x + max_x) / 2) + 1;
	int y_val = ((min_y + max_y) / 2) + 1;

	size_t crop_img_bytes = (new_height * new_width) * sizeof(pixel);

	pixel* crop_img;
	int count = 0;

	crop_img = (pixel*)malloc(crop_img_bytes);

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (c_mat[i * width + j].x >= min_x && c_mat[i * width + j].x <= max_x &&
				c_mat[i * width + j].y >= min_y && c_mat[i * width + j].y <= max_y) {
				crop_img[count].x = c_mat[i * width + j].x;
				crop_img[count].y = c_mat[i * width + j].y;
				crop_img[count].r = c_mat[i * width + j].r;
				crop_img[count].g = c_mat[i * width + j].g;
				crop_img[count].b = c_mat[i * width + j].b;
				crop_img[count].is_edge = c_mat[i * width + j].is_edge;
				crop_img[count].kmean_group = c_mat[i * width + j].kmean_group;

				count++;
			}
		}
	}

	int total = 0;
	CImg<float> image(new_width, new_height, 1, 3, 0);

	for (int i = 0; i < new_height; i++) {
		for (int j = 0; j < new_width; j++) {
			/*if (img.mat[i][j].x >= min_x && img.mat[i][j].x <= max_x &&
				img.mat[i][j].y >= min_y && img.mat[i][j].y <= max_y) {
				*/
				float orig_group[3];
				orig_group[0] = crop_img[i * new_width + j].r;
				orig_group[1] = crop_img[i* new_width + j].g;
				orig_group[2] = crop_img[i* new_width + j].b;
		

				image.draw_point(j, i, orig_group);
				total++;
				
			//}
		}
	}

	image.normalize(0, 255);

	std::string result;

	std::string name1 = "classify/cropped";
	std::string name2 = ".bmp";

	char numstr[21]; // enough to hold all numbers up to 64-bits
	char iterator_a[21];
	sprintf(numstr, "%d", kmean_grp);
	sprintf(iterator_a, "%d", iterator);
	result = name1 + numstr + iterator_a + name2;

	const char *file_name = result.c_str();

	image.save(file_name);
	
	*crop_height = new_height;
	*crop_width = new_width;

	feature_info feature_details;
	feature_details.height = new_height;
	feature_details.width = new_width;
	feature_details.img_path = file_name;
	feature_details.x = x_val;
	feature_details.y = y_val;

	save_feature_info(feature_details);

	return crop_img;
}

void gray_img_save(int width,int height,pixel* h_array_img) {

	CImg<float> image_gray(width, height, 1, 3, 0);

	int k = 10;
	while (k < 255) {
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {

				if (h_array_img[i * width + j].g <= k) {
					float orig_group0[3];
					orig_group0[0] = 0;
					orig_group0[1] = 0;
					orig_group0[2] = 0;

					image_gray.draw_point(j, i, orig_group0);

				}
				else {
					float orig_group255[3];
					orig_group255[0] = 255;
					orig_group255[1] = 255;
					orig_group255[2] = 255;

					image_gray.draw_point(j, i, orig_group255);
				}
			}
		}
		image_gray.normalize(0, 255);
		std::string result;
		std::string name1 = "thresholded/threshold_step_";
		char numstr[21]; // enough to hold all numbers up to 64-bits
		sprintf(numstr, "%d", k);
		result = name1 + numstr + ".bmp";

		const char *file_name1 = result.c_str();

		image_gray.save(file_name1);
		k = k + 10;
	}

}

void save_feature_info(feature_info feature_details){
	fstream detail_file;
	
	detail_file.open("feature_info.txt", ios_base::app);

	for (int i = 0; i < 5;i++){
		switch (i)
		{
		case 0:
			detail_file << feature_details.img_path << endl;
			break;
		case 1:
			detail_file << feature_details.height << endl;
			break;
		case 2:
			detail_file << feature_details.width << endl;
			break;
		case 3:
			detail_file << feature_details.x << endl;
			break;
		case 4:
			detail_file << feature_details.y << endl;
			break;
		default:
			break;
		}
	}


}