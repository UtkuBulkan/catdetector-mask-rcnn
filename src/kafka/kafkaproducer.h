/*
 * kafkaproducer.h
 *
 *  Created on: 2 Sep 2019
 *      Author: root
 */

#ifndef SRC_KAFKAPRODUCER_H_
#define SRC_KAFKAPRODUCER_H_

#include <string>

void video_analyser_kafka_producer(std::string brokers, std::string topic_str);

#endif /* SRC_KAFKAPRODUCER_H_ */
