//
// Created by fabri on 30/11/2025.
//

#ifndef PONG_AI_PATTERNCLASSIFIER_H
#define PONG_AI_PATTERNCLASSIFIER_H


#pragma once
#include "../nn/neural_network.h"

class PatternClassifier {
public:
    PatternClassifier();
    void train(size_t epochs = 2000);
    void test();
private:
    utec::neural_network::NeuralNetwork<double> net;
};


#endif 