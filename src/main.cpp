#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <mutex>
#include <string>
#include <cstdlib>

extern "C"
{
	#include <libswscale/swscale.h>
	#include <libavcodec/avcodec.h>
	#include <libavutil/avutil.h>
	#include <libavformat/avformat.h>
}
//onnxruntime
#include <onnxruntime_cxx_api.h>

#include "yolov5s.hpp"
#include "resnet50.hpp"
#include "common.hpp"

using namespace std;
using namespace cv;
using namespace Ort;




static const char* cocolabels[] = {
	"person", "bicycle", "car", "motorcycle", "airplane",
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
	"scissors", "teddy bear", "hair drier", "toothbrush"
};



void draw_boxes(cv::Mat frame, const std::vector<BoxInfo>& boxes) {

	for (auto& box : boxes) {
		uint8_t b, g, r;
		tie(b, g, r) = random_color(box.label);
		rectangle(frame, cv::Point(box.x1, box.y1), cv::Point(box.x2, box.y2), Scalar(b, g, r), 10);
		putText(frame, i_format("%s", cocolabels[box.label]), cv::Point(box.x1, box.y1 - 10), 0, 1, Scalar(b, g, r), 2, 16);
	}
}


int main() {

	
	YOLOv5 yolo_model = YOLOv5("yolov5s-sim-7.onnx");

	cv::Mat frame = cv::imread("bus.jpg");
	std::vector<BoxInfo> boxes;
	yolo_model.detect(frame, boxes);
	draw_boxes(frame, boxes);
	cv::imwrite("yolo_result.jpg", frame);


	// Resnet50 resnet("resnet.onnx");
	// cv::Mat img = cv::imread("test.png");
	// resnet.inference(img);

/*
	VideoCapture cam;
	string video_path = "test.mp4";
	if (cam.isOpened()){
		cout << video_path << " VideoCapture open success" << endl;
		int m_count = cam.get(CV_CAP_PROP_FRAME_COUNT);
		cv::Mat frame;
		std::vector<BoxInfo> boxes;
		while(--m_count){
			cam >> frame;
			if(frame.empty()){
				std::cout << "get empty image..." << std::endl;
				continue;
			}
				
			// cvtColor(frame, frame, CV_BGR2RGB);
			yolo_model.detect(frame, boxes);
			draw_boxes(frame, boxes);
			boxes.clear();
			cv::imshow("result.jpg", frame);
			cv::waitKey(10);
		}
	}
	else{
		throw logic_error("VideoCapture open failed");
	}
*/
	

	return 0;
}