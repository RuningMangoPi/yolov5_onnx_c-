#pragma once

#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
//onnxruntime
#include <onnxruntime_cxx_api.h>


using namespace std;
using namespace Ort;



class Resnet50 {
public:
	Resnet50(const string& model_path);
	~Resnet50();
	string inference(const cv::Mat& image);
	

private:
	float confThreshold_ = 0.6;
	int inpWidth_ = 112;
	int inpHeight_ = 112;
	int nout;
	int num_classes=10;
	string classes[10] = { "1", "2","3" };

	const bool keep_ratio = false;
	vector<float> input_image_;
	void normalize(cv::Mat img);
	cv::Mat resize_image(cv::Mat srcimg, int *newh, int *neww, int *top, int *left);

	Env env = Env(ORT_LOGGING_LEVEL_ERROR, "resnet50");

	Session *ort_session = nullptr;
	SessionOptions sessionOptions = SessionOptions();
	vector<char*> input_names;  
	vector<char*> output_names; 
	vector<vector<int64_t>> input_node_dims; 
	vector<vector<int64_t>> output_node_dims;
	int64_t result_{ 0 };
};




