#pragma once

#include "../utec/nn/neural_network.h"
#include "../utec/nn/nn_dense.h"
#include "../utec/nn/nn_activation.h"
#include "../utec/nn/nn_loss.h"
#include "../utec/nn/nn_optimizer.h"

using namespace utec::algebra;
using namespace utec::neural_network;

class SequencePredictor {
public:
    using Net = NeuralNetwork<double>;

    // Constructor
    SequencePredictor(size_t input_dimension) {
        net_.add_layer(std::make_unique<Dense<double>>(input_dimension, 6, 0.1, 0.0));
        net_.add_layer(std::make_unique<ReLU<double>>());
        net_.add_layer(std::make_unique<Dense<double>>(6, 1, 0.1, 0.0));
    }

    // Entrenamiento
    void train(const Tensor<double,2>& X, const Tensor<double,2>& Y) {
        net_.train<MSELoss, SGD>(X, Y, 800, 4, 0.005);
    }

    // Predicción del siguiente valor
    double predict_next(const Tensor<double,2>& last_values) {
        auto output = net_.predict(last_values);
        return output(0,0);
    }

private:
    Net net_;
};
