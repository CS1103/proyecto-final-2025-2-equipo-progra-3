//
// Created by fabri on 1/12/2025.
//

#include <iostream>
#include "../include/utec/nn/neural_network.h"
#include "../include/utec/nn/nn_dense.h"
#include "../include/utec/nn/nn_activation.h"
#include "../include/utec/nn/nn_loss.h"
#include "../include/utec/nn/nn_optimizer.h"
#include "../include/utec/algebra/Tensor.h"

using namespace utec::algebra;
using namespace utec::neural_network;

int main() {
    // Red simple 3->4->2
    NeuralNetwork<double> net;
    net.add_layer(std::make_unique<Dense<double>>(3, 4, 0.1, 0.0));
    net.add_layer(std::make_unique<ReLU<double>>());
    net.add_layer(std::make_unique<Dense<double>>(4, 2, 0.1, 0.0));
    net.add_layer(std::make_unique<Sigmoid<double>>());

    Tensor<double,2> X(3,3);
    X = {0.1,0.2,0.3,
         0.4,0.5,0.6,
         0.7,0.8,0.9};

    Tensor<double,2> Y(3,2);
    Y = {1,0,
         0,1,
         1,0};

    // Entrenar rápido
    net.train<MSELoss, SGD>(X, Y, 10, 1, 0.01);

    Tensor<double,2> output = net.predict(X);
    std::cout << "Salida de la red:" << std::endl << output << std::endl;

    return 0;
}
