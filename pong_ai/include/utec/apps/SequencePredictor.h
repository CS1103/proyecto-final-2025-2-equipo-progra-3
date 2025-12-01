//
// Created by fabri on 30/11/2025.
//

#ifndef PONG_AI_SEQUENCEPREDICTOR_H
#define PONG_AI_SEQUENCEPREDICTOR_H


#pragma once
#include "../nn/neural_network.h"

class SequencePredictor {
public:
    SequencePredictor();
    void train(size_t epochs = 3000);
    void test();
private:
    utec::neural_network::NeuralNetwork<double> net;
};



#endif 