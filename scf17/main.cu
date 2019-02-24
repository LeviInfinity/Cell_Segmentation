#include "globals.hpp"

int main(int argc, char *argv[]) {


	int iterator = 0;
	fstream config_file;
	config_file.open("config.txt");

	vector<string> file_names;
	
	string line;
	while (getline(config_file, line))
	{
		file_names.push_back(line);
	}

	for (auto file_n : file_names){


		//Reading Image
		cout << endl;
		cout << "Reading Image" << endl;
		image img = read_img(file_n);
		cout << "Done" << endl;
		cout << endl;

		pixel* h_array_img = flatten_image(img);

		/*
		cout << "Performing Edge Detection" << endl;
		h_array_img = edge_detect(img,h_array_img);
		cout << "Done" << endl;
		cout << endl;

		cout << "Performing K Means Clustering" << endl;
		auto result = kmeans(h_array_img,img,K_MAX);
		vector<int> SSE = result.first;
		vector<vector<pixel>> array_k = result.second;

		h_array_img = elbow_method(SSE,array_k,K_MAX);

		cout << "Done" << endl;
		cout << endl;

		cout << "Writing Image" << endl;

		pixel *cropped_img;
		int cropped_height;
		int cropped_width;

		for (int i = 0; i < K_MAX; i++){
		cropped_img = crop_kmeans_img_carray(h_array_img, img.width, img.height, i, &cropped_width, &cropped_height, img);
		}
		cout << "Done" << endl;

		*/

		//Save Thresholded Image
		gray_img_save(img.width, img.height, h_array_img);

		//Reading Image
		cout << endl;
		cout << "Reading Image Again" << endl;
		image img_1 = read_img("thresholded/threshold_step_110.bmp");
		cout << "Done" << endl;
		cout << endl;

		pixel* h_array_img_1 = flatten_image(img_1);

		cout << "Performing Edge Detection" << endl;
		h_array_img_1 = edge_detect(img, h_array_img_1);
		cout << "Done" << endl;
		cout << endl;

		cout << "Performing K Means Clustering" << endl;
		auto result = kmeans(h_array_img_1, img_1, K_MAX);
		vector<int> SSE = result.first;
		vector<vector<pixel>> array_k = result.second;

		h_array_img_1 = elbow_method(SSE, array_k, K_MAX);

		cout << "Done" << endl;

		cout << "Writing Image" << endl;

		pixel *cropped_img;
		int cropped_height;
		int cropped_width;

		for (int i = 0; i < 3; i++){
			cropped_img = crop_kmeans_img_carray(h_array_img_1, img_1.width, img_1.height, i, &cropped_width, &cropped_height, img_1,iterator);
		}

		cout << "Done" << endl;

		cout << endl;
		iterator++;
	}

	system("pause");
	return 0;
}
