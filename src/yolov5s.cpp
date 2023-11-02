#include "yolov5s.hpp"



YOLOv5::YOLOv5(const string& model_path)
{
	Configuration config = { 0.3, 0.5, 0.3, model_path};
	this->confThreshold = config.confThreshold;
	this->nmsThreshold = config.nmsThreshold;
	this->objThreshold = config.objThreshold;
	this->num_classes = 80;//sizeof(this->classes)/sizeof(this->classes[0]);  // �������
	this->inpHeight = 640;
	this->inpWidth = 640;


	sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);  //����ͼ�Ż�����

#ifdef _WIN32
  	const wchar_t* model_path_ = model_path.c_str();
#else
  	const char* model_path_ = model_path.c_str();
	
#endif

	ort_session = new Session(env, model_path_, sessionOptions);

	size_t numInputNodes = ort_session->GetInputCount();  //��������ڵ�����                         
	size_t numOutputNodes = ort_session->GetOutputCount();
	AllocatorWithDefaultOptions allocator;   // ������������ڵ��ڴ�
	for (int i = 0; i < numInputNodes; i++)
	{
		input_names.push_back(ort_session->GetInputName(i, allocator));		// �ڴ�
		Ort::TypeInfo input_type_info = ort_session->GetInputTypeInfo(i);   // ����
		auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();  // 
		auto input_dims = input_tensor_info.GetShape();    // ����shape
		input_node_dims.push_back(input_dims);	// ����
	}
	for (int i = 0; i < numOutputNodes; i++)
	{
		output_names.push_back(ort_session->GetOutputName(i, allocator));
		Ort::TypeInfo output_type_info = ort_session->GetOutputTypeInfo(i);
		auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
		auto output_dims = output_tensor_info.GetShape();
		output_node_dims.push_back(output_dims);
	}
	this->inpHeight = input_node_dims[0][2];
	this->inpWidth = input_node_dims[0][3];
	this->nout = output_node_dims[0][2];      // 5+classes
	this->num_proposal = output_node_dims[0][1];  // pre_box

}


cv::Mat YOLOv5::resize_image(cv::Mat srcimg, int *newh, int *neww, int *top, int *left)  //�޸�ͼƬ��С�����߽��ֹʧ��
{
	int srch = srcimg.rows, srcw = srcimg.cols;
	*newh = this->inpHeight;
	*neww = this->inpWidth;
	cv::Mat dstimg;
	if (this->keep_ratio && srch != srcw) {
		float hw_scale = (float)srch / srcw;
		if (hw_scale > 1) {
			*newh = this->inpHeight;
			*neww = int(this->inpWidth / hw_scale);
			resize(srcimg, dstimg, cv::Size(*neww, *newh), cv::INTER_AREA);
			*left = int((this->inpWidth - *neww) * 0.5);
			copyMakeBorder(dstimg, dstimg, 0, 0, *left, this->inpWidth - *neww - *left, cv::BORDER_CONSTANT, 114);
		}
		else {
			*newh = (int)this->inpHeight * hw_scale;
			*neww = this->inpWidth;
			resize(srcimg, dstimg, cv::Size(*neww, *newh), cv::INTER_AREA);  //�ȱ�����С����ֹʧ��
			*top = (int)(this->inpHeight - *newh) * 0.5;  //�ϲ�ȱʧ����
			copyMakeBorder(dstimg, dstimg, *top, this->inpHeight - *newh - *top, 0, 0, cv::BORDER_CONSTANT, 114); //�ϲ��top��С���²��ʣ�ಿ�֣����Ҳ��
		}
	}
	else {
		cv::resize(srcimg, dstimg, cv::Size(*neww, *newh), cv::INTER_AREA);
	}
	return dstimg;
}

void YOLOv5::normalize_(cv::Mat img)  //��һ��
{
	//    img.convertTo(img, CV_32F);
	//cout<<"picture size"<<img.rows<<img.cols<<img.channels()<<endl;
	int row = img.rows;
	int col = img.cols;
	this->input_image_.resize(row * col * img.channels());  // vector��С

	for (int c = 0; c < 3; c++)  // bgr
	{
		for (int i = 0; i < row; i++)  // ��
		{
			for (int j = 0; j < col; j++)  // ��
			{
				float pix = img.ptr<uchar>(i)[j * 3 + 2 - c];  // Mat���ptr������������һ�����ص��׵�ַ,2-c:��ʾrgb
				this->input_image_[c * row * col + i * col + j] = pix / 255.0; //��ÿ�����ؿ��һ����װ������
			}
		}
	}
}

void YOLOv5::nms(vector<BoxInfo>& input_boxes)
{
	sort(input_boxes.begin(), input_boxes.end(), [](BoxInfo a, BoxInfo b) { return a.score > b.score; }); // ��������
	vector<bool> remove_flags(input_boxes.size(), false);
	auto iou = [](const BoxInfo& box1, const BoxInfo& box2)
	{
		float xx1 = max(box1.x1, box2.x1);
		float yy1 = max(box1.y1, box2.y1);
		float xx2 = min(box1.x2, box2.x2);
		float yy2 = min(box1.y2, box2.y2);
		// ����
		float w = max(0.0f, xx2 - xx1 + 1);
		float h = max(0.0f, yy2 - yy1 + 1);
		float inter_area = w * h;
		// ����
		float union_area = max(0.0f, box1.x2 - box1.x1) * max(0.0f, box1.y2 - box1.y1)
			+ max(0.0f, box2.x2 - box2.x1) * max(0.0f, box2.y2 - box2.y1) - inter_area;
		return inter_area / union_area;
	};
	for (int i = 0; i < input_boxes.size(); ++i)
	{
		if (remove_flags[i]) continue;
		for (int j = i + 1; j < input_boxes.size(); ++j)
		{
			if (remove_flags[j]) continue;
			if (input_boxes[i].label == input_boxes[j].label && iou(input_boxes[i], input_boxes[j]) >= this->nmsThreshold)
			{
				remove_flags[j] = true;
			}
		}
	}
	int idx_t = 0;
	// remove_if()���� remove_if(beg, end, op) //�Ƴ�����[beg,end)��ÿһ�������ж�ʽ:op(elem)���true����Ԫ��
	input_boxes.erase(remove_if(input_boxes.begin(), input_boxes.end(), [&idx_t, &remove_flags](const BoxInfo& f) { return remove_flags[idx_t++]; }), input_boxes.end());
}

void YOLOv5::detect(cv::Mat& frame, std::vector<BoxInfo>& boxes)
{
	int newh = 0, neww = 0, padh = 0, padw = 0;
	cv::Mat dstimg = this->resize_image(frame, &newh, &neww, &padh, &padw); 
	this->normalize_(dstimg);  

	array<int64_t, 4> input_shape_{ 1, 3, this->inpHeight, this->inpWidth };  //1,3,640,640

	auto allocator_info = MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

	Value input_tensor_ = Value::CreateTensor<float>(allocator_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());

	vector<Value> ort_outputs = ort_session->Run(RunOptions{ nullptr }, &input_names[0], &input_tensor_, 1, output_names.data(), output_names.size());   // ��ʼ����
	// generate proposals
		//cout<<"ort_outputs_size"<<ort_outputs.size()<<endl;
	vector<BoxInfo> generate_boxes;  // BoxInfo
	float ratioh = (float)frame.rows / newh, ratiow = (float)frame.cols / neww;

	float* pdata = ort_outputs[0].GetTensorMutableData<float>(); // GetTensorMutableData

	for (int i = 0; i < num_proposal; ++i) // num_pre_boxes
	{
		int index = i * nout;      // prob[b*num_pred_boxes*(classes+5)]  
		float obj_conf = pdata[index + 4];  // 
		//cout<<"k"<<obj_conf<<endl;
		if (obj_conf > this->objThreshold)  // 
		{
			int class_idx = 0;
			float max_class_socre = 0;
			for (int k = 0; k < this->num_classes; ++k)
			{
				if (pdata[k + index + 5] > max_class_socre)
				{
					max_class_socre = pdata[k + index + 5];
					class_idx = k;
				}
			}
			max_class_socre *= obj_conf;   // 
			if (max_class_socre > this->confThreshold) //
			{
				//const int class_idx = classIdPoint.x;
				float cx = pdata[index];	//x
				float cy = pdata[index + 1];  //y
				float w = pdata[index + 2];  //w
				float h = pdata[index + 3];  //h
				//cout<<cx<<cy<<w<<h<<endl;
				float xmin = (cx - padw - 0.5 * w)*ratiow;
				float ymin = (cy - padh - 0.5 * h)*ratioh;
				float xmax = (cx - padw + 0.5 * w)*ratiow;
				float ymax = (cy - padh + 0.5 * h)*ratioh;
				//cout<<xmin<<ymin<<xmax<<ymax<<endl;
				generate_boxes.push_back(BoxInfo{ xmin, ymin, xmax, ymax, max_class_socre, class_idx });
			}
		}
	}

	// Perform non maximum suppression to eliminate redundant overlapping boxes with
	// lower confidences
	nms(generate_boxes);
	boxes = generate_boxes;
	/*for (size_t i = 0; i < generate_boxes.size(); ++i)
	{
		int xmin = int(generate_boxes[i].x1);
		int ymin = int(generate_boxes[i].y1);
		cv::rectangle(frame, cv::Point(xmin, ymin), cv::Point(int(generate_boxes[i].x2), int(generate_boxes[i].y2)), cv::Scalar(0, 0, 255), 2);
		string label = cv::format("%.2f", generate_boxes[i].score);
		label = this->classes[generate_boxes[i].label] + ":" + label;
		
		if ((this->classes[generate_boxes[i].label] == "person") || (this->classes[generate_boxes[i].label] == "cell phone")) {
			boxes.push_back(generate_boxes[i]);
		}
		putText(frame, label, cv::Point(xmin, ymin - 5), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 255, 0), 1);
	}*/
}















