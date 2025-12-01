#pragma once
#include "neural_network.h"
#include <vector>

class PatternClassifier {
public:
    using Net = utec::neural_network::NeuralNetwork<double>;

    PatternClassifier(size_t input_size) {
        // Arquitectura simple: Dense → ReLU → Dense → Sigmoid
        net_.add_layer(std::make_unique<utec::neural_network::Dense<double>>(input_size, 8, Init::xavier, Init::zeros));
        net_.add_layer(std::make_unique<utec::neural_network::ReLU<double>>());
        net_.add_layer(std::make_unique<utec::neural_network::Dense<double>>(8, 3, Init::xavier, Init::zeros));
        net_.add_layer(std::make_unique<utec::neural_network::Sigmoid<double>>());
    }

    void train(const Tensor<double,2>& X, const Tensor<double,2>& Y) {
        net_.train<utec::neural_network::MSELoss, utec::neural_network::SGD>(X, Y, 500, 4, 0.01);
    }

    Tensor<double,2> predict(const Tensor<double,2>& X) {
        return net_.predict(X);
    }

private:
    Net net_;
};
