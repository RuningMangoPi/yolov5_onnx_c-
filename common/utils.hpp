#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>
#include <string>
#include <future>


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


// 深拷贝
char* deepCopy(char* str) {
    int len = strlen(str);
    char* new_str = new char[len + 1];
    strcpy(new_str, str);
    return new_str;
}

template <typename T>
static void softmax(T& input) {
	float rowmax = *std::max_element(input.begin(), input.end());
	std::vector<float> y(input.size());
	float sum = 0.0f;
	for (size_t i = 0; i != input.size(); ++i) {
		sum += y[i] = std::exp(input[i] - rowmax);
	}
	for (size_t i = 0; i != input.size(); ++i) {
		input[i] = y[i] / sum;
	}
}


// tinypose 根据人体框外扩后crop出图片，输入进tinypose
cv::Rect expand_crop(const cv::Mat& src_img, const cv::Rect& box, float expand_ratio=0.3){
    int src_imgw = src_img.cols;
    int src_imgh = src_img.rows;

    int xmin = box.x;
    int ymin = box.y;
    int xmax = box.x + box.width - 1;
    int ymax = box.y + box.height - 1;

    float h_half = (ymax - ymin) * (1 + expand_ratio) / 2.;
    float w_half = (xmax - xmin) * (1 + expand_ratio) / 2.;
    if( h_half > w_half * 4 / 3)
        w_half = h_half * 0.75;

    cout << "h_half, w_half   :  " << h_half << "  " << w_half << endl;

    cv::Point center = cv::Point((ymin + ymax) / 2., (xmin + xmax) / 2.);

    ymin = max(1, int(center.x - h_half));
    ymax = min(src_imgh-1, int(center.x + h_half)-1);
    xmin = max(1, int(center.y - w_half));
    xmax = min(src_imgw-1, int(center.y + w_half)-1);

    cout << "src_imgw, src_imgh:  " << src_imgw << "  " << src_imgh << endl;

    cv::Rect new_rect = cv::Rect(xmin, ymin, xmax-xmin+1, ymax-ymin+1);



    cout << "new x1, y1, x2, y2:  " << xmin << "  " << ymin << "  " << xmax << "  " << ymax << endl;


    cv::Mat rect_image = src_img.clone()(new_rect);
    cout << "11111111111111111111111" << endl;

    return new_rect;
}