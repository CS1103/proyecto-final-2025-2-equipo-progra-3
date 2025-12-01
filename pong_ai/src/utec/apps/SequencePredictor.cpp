
#include "../../../include/utec/apps/SequencePredictor.h"
#include "../../../include/utec/algebra/tensor.h"
#include "../../../include/utec/nn/nn_dense.h"
#include "../../../include/utec/nn/nn_activation.h"
#include "../../../include/utec/nn/nn_loss.h"
#include "../../../include/utec/nn/neural_network.h"
#include <iostream>
#include <random>

SequencePredictor::SequencePredictor() {

    std::mt19937 gen(10);
    auto xavier = [&](auto& M){
        double limit = std::sqrt(6.0 / (M.shape()[0] + M.shape()[1]));
        std::uniform_real_distribution<> dist(-limit, limit);
        for(auto &v : M) v = dist(gen);
    };

    net.add_layer(std::make_unique<utec::neural_network::Dense<double>>(3, 6, xavier, xavier));
    net.add_layer(std::make_unique<utec::neural_network::ReLU<double>>());
    net.add_layer(std::make_unique<utec::neural_network::Dense<double>>(6, 1, xavier, xavier));
    net.add_layer(std::make_unique<utec::neural_network::Sigmoid<double>>());
}

void SequencePredictor::train(size_t epochs) {

    utec::algebra::Tensor<double,2> X({200, 3});
    utec::algebra::Tensor<double,2> Y({200, 1});

    for (int i = 0; i < 200; i++) {
        double a = i;
        double b = i+1;
        double c = i+2;
        double next = i+3;

        X(i,0)=a/200.0;
        X(i,1)=b/200.0;
        X(i,2)=c/200.0;
        Y(i,0)=next/200.0;
    }

    net.train<utec::neural_network::MSELoss>(X, Y, epochs, 20, 0.05);
}

void SequencePredictor::test() {
    utec::algebra::Tensor<double,2> input({1,3});
    input = {10/200.0, 11/200.0, 12/200.0};

    auto pred = net.predict(input);

    std::cout << "\nSequence prediction (expect ~13): "
              << pred(0,0) * 200.0 << "\n";
}
