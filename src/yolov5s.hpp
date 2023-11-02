#pragma once

#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
//onnxruntime
#include <onnxruntime_cxx_api.h>


using namespace std;
using namespace Ort;



struct Configuration
{
public:
	float confThreshold; // Confidence threshold
	float nmsThreshold;  // Non-maximum suppression threshold
	float objThreshold;  //Object Confidence threshold
	string modelpath;
};

typedef struct BoxInfo
{
	float x1;
	float y1;
	float x2;
	float y2;
	float score;
	int label;
} BoxInfo;




class YOLOv5
{
public:
	//YOLOv5(Configuration config);
	YOLOv5(const string& model_path);
	void detect(cv::Mat& frame, std::vector<BoxInfo>& boxes);
private:
	float confThreshold;
	float nmsThreshold;
	float objThreshold;
	int inpWidth;
	int inpHeight;
	int nout;
	int num_proposal;
	int num_classes;
	string classes[80] = { "person", "bicycle", "car", "motorcycle", "airplane",
	"bus", "train", "truck", "boat", "traffic light", "fire hydrant",
	"stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
	"sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
	"umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
	"snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
	"skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
	"cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
	"orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
	"chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv",
	"laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
	"oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
	"scissors", "teddy bear", "hair drier", "toothbrush" };


	const bool keep_ratio = false;
	vector<float> input_image_;		// ����ͼƬ
	void normalize_(cv::Mat img);		// ��һ������
	void nms(vector<BoxInfo>& input_boxes);
	cv::Mat resize_image(cv::Mat srcimg, int *newh, int *neww, int *top, int *left);

	Env env = Env(ORT_LOGGING_LEVEL_ERROR, "yolov5s-sim-7"); // ��ʼ������


	//Env *env = nullptr;

	Session *ort_session = nullptr;    // ��ʼ��Sessionָ��ѡ��
	SessionOptions sessionOptions = SessionOptions();  //��ʼ��Session����
	//SessionOptions *sessionOptions = nullptr;
	vector<char*> input_names;  // ����һ���ַ�ָ��vector
	vector<char*> output_names; // ����һ���ַ�ָ��vector
	vector<vector<int64_t>> input_node_dims; // >=1 outputs  ����άvector
	vector<vector<int64_t>> output_node_dims; // >=1 outputs ,int64_t C/C++��׼
};



