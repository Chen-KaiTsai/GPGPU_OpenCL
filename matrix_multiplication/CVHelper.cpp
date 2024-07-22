#include "main.h"

void CVHelper::CVHWC2ArrayCWH(const char* fileName, unique_ptr<float[]> data)
{
	cv::Mat image = cv::imread(fileName, cv::IMREAD_COLOR);
	cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

	cv::Mat If32;
	image.convertTo(If32, CV_32FC3);

	for (int c = 0; c < If32.channels(); ++c)
		for (int y = 0; y < image.rows; ++y)
			for (int x = 0; x < image.cols; ++x) {
				data[c * image.rows * image.cols + y * image.cols + x] = (float)If32.at<cv::Vec3f>(y, x)[c];
			}
}

void CVHelper::showImgWithArrayCWH(unique_ptr<float[]> data, int C, int H, int W)
{
	cv::Mat image(H, W, CV_8UC3);

	for (int c = 0; c < C; ++c)
		for (int y = 0; y < H; ++y)
			for (int x = 0; x < W; ++x) {
				image.at<cv::Vec3b>(y, x)[c] = static_cast<uint8_t>(data[c * H * W + y * W + x]);
			}

	cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
	cv::resize(image, image, cv::Size(W / 2, H / 2), cv::InterpolationFlags::INTER_AREA);
	cv::namedWindow("Output", cv::WINDOW_AUTOSIZE);
	cv::imshow("Output", image);
	cv::waitKey(0);
}
