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
 *
 * Some parts of this file has been inspired and influenced from this link :
 * https://www.learnopencv.com/deep-learning-based-object-detection
 * -and-instance-segmentation-using-mask-r-cnn-in-opencv-python-c/
 *
 */
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <syslog.h>

#include "tensorflow/c/c_api.h"
#include "objectdetection.h"
#include "md5_helper.h"

#define CATDETECTOR_ANALYSE_EVERY_24_FRAMES
#define CATDETECTOR_ENABLE_OUTPUT_TO_VIDEO_FILE
#define CATDETECTOR_ENABLE_CAPTURED_FRAMES_TO_JSON

ObjectDetector::ObjectDetector() : confidence_threshold(0.9),
mask_threshold(0.3),
class_definition_file("../data/mscoco_labels.names"),
colors_file("../data/colors.txt"),
text_graph_file("../data/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"),
model_weights_file("../data/frozen_inference_graph.pb")
{
	syslog (LOG_NOTICE, "ObjectDetector Constructor Begin");

	std::ifstream classes_file_stream(class_definition_file.c_str());
	std::ifstream colors_file_stream(colors_file.c_str());
	std::string line;

	while (getline(classes_file_stream, line)) {
		classes.push_back(line);
		syslog (LOG_NOTICE, "Class Labels : %s", line.c_str());
	}

	while (getline(colors_file_stream, line)) {
		std::stringstream ss(line);
		double red, green, blue;
		ss >> red >> green >> blue;
		colors.push_back(cv::Scalar(red, green, blue, 255.0));
		syslog (LOG_NOTICE, "Colors.txt Colors : %f, %f, %f", red, green, blue);
	}

	// Load the network for the model
	syslog (LOG_NOTICE, "ObjectDetector Loading Network");
	net = cv::dnn::readNetFromTensorflow(model_weights_file, text_graph_file);
	net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
	net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
	syslog (LOG_NOTICE, "ObjectDetector Network Loaded");

	syslog (LOG_NOTICE, "ObjectDetector Constructor End");
}

ObjectDetector::~ObjectDetector() {
}

void ObjectDetector::draw_box(cv::Mat& frame, int classId, float confidence, cv::Rect box, cv::Mat& objectMask)
{
	syslog(LOG_NOTICE, "ObjectDetector::draw_box Begin");

	std::string label = cv::format("%2.2f", confidence);
	std::vector<cv::Mat> contours;
	cv::Mat hierarchy, mask, coloredRoi;
	cv::Size labelSize;
	cv::Scalar color;
	int baseLine;

	cv::rectangle(frame, cv::Point(box.x, box.y), cv::Point(box.x+box.width, box.y+box.height),cv::Scalar(255, 178, 50), 3);
	if (!classes.empty())
	{
		CV_Assert(classId < (int)classes.size());
		label = classes[classId] + ":" + label;
	}
	labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
	box.y = cv::max(box.y, labelSize.height);
	cv::rectangle(frame, cv::Point(box.x, box.y - round(1.5*labelSize.height)), cv::Point(box.x + round(1.5*labelSize.width), box.y + baseLine), cv::Scalar(255, 255, 255), cv::FILLED);
	cv::putText(frame, label, cv::Point(box.x, box.y), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0,0,0),1);
	color = colors[classId%colors.size()];

	cv::resize(objectMask, objectMask, cv::Size(box.width, box.height));
	mask = (objectMask > mask_threshold);
	coloredRoi = (0.3 * color + 0.7 * frame(box));
	coloredRoi.convertTo(coloredRoi, CV_8UC3);

	mask.convertTo(mask, CV_8U);
	cv::findContours(mask, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);
	cv::drawContours(coloredRoi, contours, -1, color, 5, cv::LINE_8, hierarchy, 100);
	coloredRoi.copyTo(frame(box), mask);

	syslog(LOG_NOTICE, "ObjectDetector::draw_box End");
}

void ObjectDetector::generate_json(cv::Mat &frame, const int &classId, const int &framecount, std::string frame_md5, std::string video_md5)
{
	syslog(LOG_NOTICE, "ObjectDetector::generate_json Begin");

	json local_j;

	local_j["class"] = classes[classId].c_str();
	local_j["frame"] = framecount;
	local_j["hash-frame"] = frame_md5;
	local_j["hash-video"] = video_md5;

	std::vector<uchar> buffer;
#define MB 1024 * 1024
	buffer.resize(200*MB);
	cv::imencode(".png", frame, buffer);
	local_j["image"] = buffer;
	j.push_back(local_j);

	syslog(LOG_NOTICE, "ObjectDetector::generate_json End");
}

void ObjectDetector::post_process(cv::Mat& frame, const std::vector<cv::Mat>& outs, int framecount, std::string hash_video)
{
	syslog(LOG_NOTICE, "ObjectDetector::post_process Begin");

	cv::Mat output_detections = outs[0], outMasks = outs[1];
	const int num_detections = output_detections.size[2];
	const int num_classes = outMasks.size[1];

	output_detections = output_detections.reshape(1, output_detections.total() / 7);
	syslog(LOG_NOTICE, "Object Detector postprocess num_detections : %d", num_detections);
	syslog(LOG_NOTICE, "Object Detector postprocess num_classes : %d", num_classes);
	for (int i = 0; i < num_detections; ++i)
	{
		float score = output_detections.at<float>(i, 2);
		if (score > confidence_threshold)
		{
			int classId = static_cast<int>(output_detections.at<float>(i, 1));
			int left = static_cast<int>(frame.cols * output_detections.at<float>(i, 3));
			int top = static_cast<int>(frame.rows * output_detections.at<float>(i, 4));
			int right = static_cast<int>(frame.cols * output_detections.at<float>(i, 5));
			int bottom = static_cast<int>(frame.rows * output_detections.at<float>(i, 6));

			left = cv::max(0, cv::min(left, frame.cols - 1));
			top = cv::max(0, cv::min(top, frame.rows - 1));
			right = cv::max(0, cv::min(right, frame.cols - 1));
			bottom = cv::max(0, cv::min(bottom, frame.rows - 1));
			cv::Rect box = cv::Rect(left, top, right - left + 1, bottom - top + 1);
			cv::Mat objectMask(outMasks.size[2], outMasks.size[3],CV_32F, outMasks.ptr<float>(i,classId));
			draw_box(frame, classId, score, box, objectMask);

			generate_json(frame, classId, framecount, "", hash_video);
		}
	}

	syslog(LOG_NOTICE, "ObjectDetector::post_process End");
}

void ObjectDetector::process_frame(cv::Mat &frame, int framecount, std::string hash_video) {
	std::vector<std::string> outNames(2);
	std::vector<double> layersTimes;
	std::vector<cv::Mat> outs;
	std::string label;
	cv::Mat blob;
	double freq, t;

	syslog(LOG_NOTICE, "ObjectDetector::process_frame Begin");

	//cv::dnn::blobFromImage(frame, blob);
	cv::dnn::blobFromImage(frame, blob, 1.0, cv::Size(frame.cols, frame.rows), cv::Scalar(), true, false);
	net.setInput(blob);
	outNames[0] = "detection_out_final";
	outNames[1] = "detection_masks";
	net.forward(outs, outNames);
	syslog(LOG_NOTICE, "Number of outs : %d", (int) outs.size());

	post_process(frame, outs, framecount, hash_video);
	freq = cv::getTickFrequency() / 1000;
	t = net.getPerfProfile(layersTimes) / freq;
	label = cv::format("London South Bank University - Utku Bulkan - Frame processing time: %.2f ms", t);
	cv::putText(frame, label, cv::Point(0, 20), cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 0, 0));

	syslog(LOG_NOTICE, "ObjectDetector::process_frame End");
}

void ObjectDetector::loop() {
	cv::Mat frame;
	cv::VideoCapture capture;
	cv::VideoWriter outputVideo;

	syslog(LOG_NOTICE, "Hello from TensorFlow C library version : %s", TF_Version());
	syslog(LOG_NOTICE, "Opening file : %s", filename.c_str());

	capture.open(filename);
	if ( !capture.isOpened	() ) {
		throw "Error opening file.\n";
	}

	capture >> frame;

	int ex = static_cast<int>(capture.get(cv::CAP_PROP_FOURCC));	// Get Codec Type- Int form
	int codec = cv::VideoWriter::fourcc('M', 'P', 'G', '2');
	cv::Size S = cv::Size((int) capture.get(cv::CAP_PROP_FRAME_WIDTH), (int) capture.get(cv::CAP_PROP_FRAME_HEIGHT));

	syslog(LOG_NOTICE, "Input file fourcc: %d, %d", codec, ex);
	syslog(LOG_NOTICE, "Input file width: %d", S.width);
	syslog(LOG_NOTICE, "Input file height: %d", S.height);

	outputVideo.open("./output.mp4", cv::CAP_FFMPEG, codec, capture.get(cv::CAP_PROP_FPS), S, true);
	outputVideo << frame;

	cv::namedWindow("Camera1", cv::WINDOW_NORMAL);
	cv::resizeWindow("Camera1", 640, 480);

	int framecount = 0;
	std::string hash_video = md5_hash(filename);

	while(1) {
		syslog(LOG_NOTICE, "Frame count : %d", framecount);
		syslog(LOG_NOTICE, "Frame resolution : %d x %d", frame.rows, frame.cols);

		capture >> frame;
		framecount++;
#ifdef CATDETECTOR_ANALYSE_EVERY_24_FRAMES
		if (framecount % 24 == 0)
#endif
		{
			process_frame(frame, framecount, hash_video);
#ifdef CATDETECTOR_ENABLE_OUTPUT_TO_VIDEO_FILE
			/* Outputting captured frames to a video file */
			outputVideo << frame;
#endif
			cv::imshow("Camera1", frame);

#ifdef CATDETECTOR_ENABLE_CAPTURED_FRAMES_TO_JSON
			/* Outputting captured frames to json */
			std::ofstream myfile;
			std::string videodata_filename(hash_video + ".json");
			myfile.open (videodata_filename);
			myfile << j << std::endl;
			myfile.close();
#endif
			/* Sending the data as a Kafka producer */
			/* video_analyser_kafka_producer(j.dump().c_str(), "TutorialTopic"); */
		}
		if(cv::waitKey(30) >= 0) break;
	}
	outputVideo.release();
}
