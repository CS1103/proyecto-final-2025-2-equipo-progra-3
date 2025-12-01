#pragma once
#include "neural_network.h"

class SequencePredictor {
public:
    using Net = utec::neural_network::NeuralNetwork<double>;

    SequencePredictor(size_t input_dimension) {
        net_.add_layer(std::make_unique<Dense<double>>(input_dimension, 6, Init::xavier, Init::zeros));
        net_.add_layer(std::make_unique<ReLU<double>>());
        net_.add_layer(std::make_unique<Dense<double>>(6, 1, Init::xavier, Init::zeros));
    }

    void train(const Tensor<double,2>& X, const Tensor<double,2>& Y) {
        net_.train<MSELoss, SGD>(X, Y, 800, 4, 0.005);
    }

    double predict_next(const Tensor<double,2>& last_values) {
        auto output = net_.predict(last_values);
        return output(0,0);
    }

private:
    Net net_;
};
