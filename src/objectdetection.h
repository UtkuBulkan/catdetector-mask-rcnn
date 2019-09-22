/*
 * Copyright 2018 SU Technology Ltd. All rights reserved.
 *
 * MIT License
 *
 * Copyright (c) 2018 SU Technology
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.

 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
*/
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <string>
#include "json.hpp"

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

using json = nlohmann::json;

class ObjectDetector {
public:
	ObjectDetector();
	~ObjectDetector();
	void process_frame(cv::Mat &frame, int framecount, std::string hash_video);
	void draw_box(cv::Mat& frame, int classId, float conf, cv::Rect box, cv::Mat& objectMask);
	void post_process(cv::Mat& frame, const std::vector<cv::Mat>& outs, int framecount, std::string hash_video);
	void loop();
	void generate_json(cv::Mat &frame, const int &classId, const int &framecount, std::string frame_md5, std::string video_md5);

	std::string filename;
private:
	// Initialize the parameters
	float confidence_threshold; // Confidence threshold
	float mask_threshold; // Mask threshold
	cv::dnn::Net net;
	std::vector < std::string > classes;
	std::vector<cv::Scalar> colors;
	// Give the configuration and weight files for the model
	std::string class_definition_file;
	std::string colors_file;
	std::string text_graph_file;
	std::string model_weights_file;

	json j;
};
