//
// Created by fabri on 30/11/2025.
//

#ifndef PONG_AI_CONTROLLERDEMO_H
#define PONG_AI_CONTROLLERDEMO_H

#pragma once
#include "../nn/neural_network.h"

class ControllerDemo {
public:
    ControllerDemo();
    void train(size_t epochs = 2500);
    void test();
private:
    utec::neural_network::NeuralNetwork<double> net;
};


#endif 