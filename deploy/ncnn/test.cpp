#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "net.h"
#include <opencv2/opencv.hpp>

struct Box {
	float x1;
	float y1;
	float x2;
	float y2;
	int cls;
	float score;
};

static float sigmoid(float x)
{
	return 1.f / (1.f + std::exp(-x));
}

static float iou(const Box &a, const Box &b)
{
	const float xx1 = std::max(a.x1, b.x1);
	const float yy1 = std::max(a.y1, b.y1);
	const float xx2 = std::min(a.x2, b.x2);
	const float yy2 = std::min(a.y2, b.y2);
	const float w = std::max(0.f, xx2 - xx1);
	const float h = std::max(0.f, yy2 - yy1);
	const float inter = w * h;
	const float area_a = std::max(0.f, a.x2 - a.x1) * std::max(0.f, a.y2 - a.y1);
	const float area_b = std::max(0.f, b.x2 - b.x1) * std::max(0.f, b.y2 - b.y1);
	const float uni = area_a + area_b - inter;

	return uni > 0.f ? inter / uni : 0.f;
}

static std::vector<Box> nms(std::vector<Box> boxes, float iou_thres)
{
	std::sort(boxes.begin(), boxes.end(), [](const Box &a, const Box &b) {
		return a.score > b.score;
	});

	std::vector<Box> keep;
	std::vector<char> removed(boxes.size(), 0);

	for (size_t i = 0; i < boxes.size(); ++i) {
		if (removed[i])
			continue;
		keep.push_back(boxes[i]);

		for (size_t j = i + 1; j < boxes.size(); ++j) {
			if (removed[j])
				continue;
			if (boxes[i].cls != boxes[j].cls)
				continue;
			if (iou(boxes[i], boxes[j]) > iou_thres)
				removed[j] = 1;
		}
	}

	return keep;
}

int main(int argc, char **argv)
{
	const char *param = "fastestdetv2.param";
	const char *bin = "fastestdetv2.ncnn.bin";
	const char *input_blob = "in0";
	const char *output_blob = "out0";
	const char *image_path = "input.jpg";
	const char *out_path = "output.jpg";
    // ./test [IN_IMG] [OUTPUT_IMG]
	if (argc > 1)
		image_path = argv[1];
	if (argc > 2)
		out_path = argv[2];
	const int input_width = 352;
	const int input_height = 352;
	const int num_classes = 80;
	const float conf_thres = 0.65f;
	const float iou_thres = 0.45f;
	static const char *class_names[] = {
		"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
		"fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
		"elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
		"skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
		"tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
		"sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
		"potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
		"microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
		"hair drier", "toothbrush"
	};

	ncnn::Net net;
	if (net.load_param(param) != 0 || net.load_model(bin) != 0) {
		std::printf("Failed to load model: %s / %s\n", param, bin);
		return -1;
	}
	cv::Mat image = cv::imread(image_path);
	if (image.empty()) {
		std::printf("Failed to read image: %s\n", image_path);
		return -1;
	}

	ncnn::Mat input = ncnn::Mat::from_pixels_resize(
		image.data, ncnn::Mat::PIXEL_BGR, image.cols, image.rows, input_width, input_height);
	const float norm_vals[3] = { 1.f / 255.f, 1.f / 255.f, 1.f / 255.f };
	input.substract_mean_normalize(0, norm_vals);
	ncnn::Extractor ex = net.create_extractor();
	ex.input(input_blob, input);
	ncnn::Mat out;
	if (ex.extract(output_blob, out) != 0) {
		std::printf("Failed to extract output blob %s\n", output_blob);
		return -1;
	}

	std::vector<Box> boxes;
	for (int y = 0; y < out.h; ++y) {
		for (int x = 0; x < out.w; ++x) {
			const float obj = out.channel(0).row(y)[x];
			int cls = 0;
			float cls_max = 0.f;
			for (int i = 0; i < num_classes; ++i) {
				const float cls_score = out.channel(5 + i).row(y)[x];
				if (cls_score > cls_max) {
					cls_max = cls_score;
					cls = i;
				}
			}
			const float score = std::pow(obj, 0.6f) * std::pow(cls_max, 0.4f);
			if (score < conf_thres)
				continue;
			const float tx = std::tanh(out.channel(1).row(y)[x]);
			const float ty = std::tanh(out.channel(2).row(y)[x]);
			const float tw = sigmoid(out.channel(3).row(y)[x]);
			const float th = sigmoid(out.channel(4).row(y)[x]);
			const float cx = (x + tx) / out.w;
			const float cy = (y + ty) / out.h;
			Box b;
			b.x1 = (cx - 0.5f * tw) * image.cols;
			b.y1 = (cy - 0.5f * th) * image.rows;
			b.x2 = (cx + 0.5f * tw) * image.cols;
			b.y2 = (cy + 0.5f * th) * image.rows;
			b.x1 = std::max(0.f, std::min(b.x1, (float)(image.cols - 1)));
			b.y1 = std::max(0.f, std::min(b.y1, (float)(image.rows - 1)));
			b.x2 = std::max(0.f, std::min(b.x2, (float)(image.cols - 1)));
			b.y2 = std::max(0.f, std::min(b.y2, (float)(image.rows - 1)));
			if (b.x2 <= b.x1 || b.y2 <= b.y1)
				continue;
			b.cls = cls;
			b.score = score;
			boxes.push_back(b);
		}
	}

	boxes = nms(boxes, iou_thres);
	for (const auto &b : boxes) {
		cv::rectangle(image,
			cv::Point((int)b.x1, (int)b.y1),
			cv::Point((int)b.x2, (int)b.y2),
			cv::Scalar(255, 255, 0), 2);
		char label[128];
		std::snprintf(label, sizeof(label), "%d %s %.2f",
			(int)b.cls, class_names[b.cls], b.score);
		cv::putText(image, label,
			cv::Point((int)b.x1, (int)b.y1 - 5),
			cv::FONT_HERSHEY_PLAIN, 0.6,
			cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
		cv::putText(image, label,
			cv::Point((int)b.x1 + 1, (int)b.y1 - 5),
			cv::FONT_HERSHEY_PLAIN, 0.6,
			cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
	}
	cv::imwrite(out_path, image);
	std::printf("Saved result to %s\n", out_path);
	return 0;
}
