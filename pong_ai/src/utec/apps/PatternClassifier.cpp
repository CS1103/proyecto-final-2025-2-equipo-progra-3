//
// Created by fabri on 30/11/2025.
//
#include "../../../include/utec/algebra/tensor.h"
#include "../../../include/utec/nn/nn_dense.h"
#include "../../../include/utec/nn/nn_activation.h"
#include "../../../include/utec/nn/nn_loss.h"
#include "../../../include/utec/nn/neural_network.h"

#include "../../../include/utec/apps/PatternClassifier.h"
#include <iostream>
#include <random>

PatternClassifier::PatternClassifier() {

    std::mt19937 gen(4);
    auto xavier = [&](auto& M){
        double limit = std::sqrt(6.0 / (M.shape()[0] + M.shape()[1]));
        std::uniform_real_distribution<> dist(-limit, limit);
        for(auto &v : M) v = dist(gen);
    };

    net.add_layer(std::make_unique<utec::neural_network::Dense<double>>(3, 4, xavier, xavier));
    net.add_layer(std::make_unique<utec::neural_network::ReLU<double>>());
    net.add_layer(std::make_unique<utec::neural_network::Dense<double>>(4, 3, xavier, xavier));
    net.add_layer(std::make_unique<utec::neural_network::Sigmoid<double>>());
}

void PatternClassifier::train(size_t epochs) {
    utec::algebra::Tensor<double,2> X({300, 3});
    utec::algebra::Tensor<double,2> Y({300, 3});

    // Círculo
    for (int i = 0; i < 100; i++) {
        X(i,0)=1; X(i,1)=0; X(i,2)=0;
        Y(i,0)=1; Y(i,1)=0; Y(i,2)=0;
    }
    // Cuadrado
    for (int i = 100; i < 200; i++) {
        X(i,0)=0; X(i,1)=1; X(i,2)=0;
        Y(i,0)=0; Y(i,1)=1; Y(i,2)=0;
    }
    // Triángulo
    for (int i = 200; i < 300; i++) {
        X(i,0)=0; X(i,1)=0; X(i,2)=1;
        Y(i,0)=0; Y(i,1)=0; Y(i,2)=1;
    }

    net.train<utec::neural_network::MSELoss>(X, Y, epochs, 32, 0.05);
}

void PatternClassifier::test() {
    utec::algebra::Tensor<double,2> test({3,3});
    test = {1,0,0,
            0,1,0,
            0,0,1};

    auto pred = net.predict(test);

    std::cout << "Pattern Classifier Predictions:\n" << pred << std::endl;
}
