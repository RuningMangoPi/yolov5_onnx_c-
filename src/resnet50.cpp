#include "resnet50.hpp"



static void softmax(vector<float>& input) {

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


Resnet50::Resnet50(const string& model_path) {
	sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
#ifdef _WIN32
  	const wchar_t* model_path = model_path.c_str();
#else
  	const char* model_path_ = model_path.c_str();
#endif
	ort_session = new Session(env, model_path_, sessionOptions);

	size_t numInputNodes = ort_session->GetInputCount();                       
	size_t numOutputNodes = ort_session->GetOutputCount();
	AllocatorWithDefaultOptions allocator;  
	for (int i = 0; i < numInputNodes; i++)
	{
		input_names.push_back(ort_session->GetInputName(i, allocator));	
		Ort::TypeInfo input_type_info = ort_session->GetInputTypeInfo(i);   
		auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();  // 
		auto input_dims = input_tensor_info.GetShape(); 
		input_node_dims.push_back(input_dims);
	}
	for (int i = 0; i < numOutputNodes; i++)
	{
		output_names.push_back(ort_session->GetOutputName(i, allocator));
		Ort::TypeInfo output_type_info = ort_session->GetOutputTypeInfo(i);
		auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
		auto output_dims = output_tensor_info.GetShape();
		output_node_dims.push_back(output_dims);
	}
	this->inpHeight_ = input_node_dims[0][2];
	this->inpWidth_ = input_node_dims[0][3];
	this->nout = output_node_dims[0][1];      // 2 classes

}


Resnet50::~Resnet50() {

}


void Resnet50::normalize(cv::Mat img)  //归一化
{
	//    img.convertTo(img, CV_32F);
	//cout<<"picture size"<<img.rows<<img.cols<<img.channels()<<endl;
	int row = img.rows;
	int col = img.cols;
	this->input_image_.resize(row * col * img.channels());  // vector大小

	for (int c = 0; c < 3; c++)  // bgr
	{
		for (int i = 0; i < row; i++)  // 行
		{
			for (int j = 0; j < col; j++)  // 列
			{
				float pix = img.ptr<uchar>(i)[j * 3 + 2 - c];  // Mat里的ptr函数访问任意一行像素的首地址,2-c:表示rgb
				this->input_image_[c * row * col + i * col + j] = pix / 255.0; //将每个像素块归一化后装进容器
			}
		}
	}
}

cv::Mat Resnet50::resize_image(cv::Mat srcimg, int *newh, int *neww, int *top, int *left) {
	int srch = srcimg.rows, srcw = srcimg.cols;
	*newh = this->inpHeight_;
	*neww = this->inpWidth_;
	cv::Mat dstimg;
	if (this->keep_ratio && srch != srcw) {
		float hw_scale = (float)srch / srcw;
		if (hw_scale > 1) {
			*newh = this->inpHeight_;
			*neww = int(this->inpWidth_ / hw_scale);
			cv::resize(srcimg, dstimg, cv::Size(*neww, *newh), cv::INTER_AREA);
			*left = int((this->inpWidth_ - *neww) * 0.5);
			cv::copyMakeBorder(dstimg, dstimg, 0, 0, *left, this->inpWidth_ - *neww - *left, cv::BORDER_CONSTANT, 114);
		}
		else {
			*newh = (int)this->inpHeight_ * hw_scale;
			*neww = this->inpWidth_;
			cv::resize(srcimg, dstimg, cv::Size(*neww, *newh), cv::INTER_AREA);  //等比例缩小，防止失真
			*top = (int)(this->inpHeight_ - *newh) * 0.5;  //上部缺失部分
			cv::copyMakeBorder(dstimg, dstimg, *top, this->inpHeight_ - *newh - *top, 0, 0, cv::BORDER_CONSTANT, 114); //上部填补top大小，下部填补剩余部分，左右不填补
		}
	}
	else {
		cv::resize(srcimg, dstimg, cv::Size(*neww, *newh), cv::INTER_AREA);
	}
	return dstimg;
}

string Resnet50::inference(const cv::Mat& image) {

	cv::Mat frame = image.clone();

	int newh = 0, neww = 0, padh = 0, padw = 0;
	cv::Mat dstimg = this->resize_image(frame, &newh, &neww, &padh, &padw);   //改大小后做padding防失真
	this->normalize(dstimg);       //归一化
	// 定义一个输入矩阵，int64_t是下面作为输入参数时的类型
	array<int64_t, 4> input_shape_{ 1, 3, this->inpHeight_, this->inpWidth_ };  //1,3,112,112
	
	//创建输入tensor
	/*
	这一行代码的作用是创建一个指向CPU内存的分配器信息对象(AllocatorInfo)，用于在运行时分配和释放CPU内存。
	它调用了CreateCpu函数并传递两个参数：OrtDeviceAllocator和OrtMemTypeCPU。
	其中，OrtDeviceAllocator表示使用默认的设备分配器，OrtMemTypeCPU表示在CPU上分配内存。
	通过这个对象，我们可以在运行时为张量分配内存，并且可以保证这些内存在计算完成后被正确地释放，避免内存泄漏的问题。
	*/
	auto allocator_info = MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	//使用Ort库创建一个输入张量，其中包含了需要进行目标检测的图像数据。
	Value input_tensor_ = Value::CreateTensor<float>(allocator_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());
	// 开始推理
	vector<Value> ort_outputs = ort_session->Run(RunOptions{ nullptr }, &input_names[0], &input_tensor_, 1, output_names.data(), output_names.size());   // 开始推理
	
	float* pdata = ort_outputs[0].GetTensorMutableData<float>();

	vector<float> results(num_classes);

	for (int i = 0; i < num_classes; ++i) {
		results[i] = pdata[i];
	}
	softmax(results);
	//softmax(pdata);
	result_ = std::max_element(results.begin(), results.end()) - results.begin();
	

	cout << result_ << endl;

	return "";
}
