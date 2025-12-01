#pragma once

#include "../utec/nn/neural_network.h"
#include "../utec/nn/nn_dense.h"
#include "../utec/nn/nn_activation.h"
#include "../utec/nn/nn_loss.h"
#include "../utec/nn/nn_optimizer.h"

using namespace utec::algebra;         // para Tensor
using namespace utec::neural_network;  // para Dense, ReLU, Sigmoid, MSELoss, SGD

class PatternClassifier {
public:
    using Net = NeuralNetwork<double>;

    // Constructor
    PatternClassifier(size_t input_size) {
        net_.add_layer(std::make_unique<Dense<double>>(input_size, 8, 0.1, 0.0));
        net_.add_layer(std::make_unique<ReLU<double>>());
        net_.add_layer(std::make_unique<Dense<double>>(8, 3, 0.1, 0.0));
        net_.add_layer(std::make_unique<Sigmoid<double>>());
    }

    // Entrenamiento
    void train(const Tensor<double,2>& X, const Tensor<double,2>& Y) {
        net_.train<MSELoss, SGD>(X, Y, 500, 4, 0.01);
    }

    // Predicción
    Tensor<double,2> predict(const Tensor<double,2>& X) {
        return net_.predict(X);
    }

private:
    Net net_;
};
