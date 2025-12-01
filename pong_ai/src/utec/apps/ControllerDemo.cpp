
#include "../../../include/utec/apps/ControllerDemo.h"
#include "../../../include/utec/algebra/tensor.h"
#include "../../../include/utec/nn/nn_dense.h"
#include "../../../include/utec/nn/nn_activation.h"
#include "../../../include/utec/nn/nn_loss.h"
#include "../../../include/utec/nn/neural_network.h"
#include <iostream>
#include <random>
#include <cmath>

ControllerDemo::ControllerDemo() {

    std::mt19937 gen(20);
    auto xavier = [&](auto& M){
        double limit = std::sqrt(6.0 / (M.shape()[0] + M.shape()[1]));
        std::uniform_real_distribution<> dist(-limit, limit);
        for(auto &v : M) v = dist(gen);
    };

    net.add_layer(std::make_unique<utec::neural_network::Dense<double>>(2, 6, xavier, xavier));
    net.add_layer(std::make_unique<utec::neural_network::ReLU<double>>());
    net.add_layer(std::make_unique<utec::neural_network::Dense<double>>(6, 1, xavier, xavier));
    net.add_layer(std::make_unique<utec::neural_network::Sigmoid<double>>());
}

void ControllerDemo::train(size_t epochs) {

    utec::algebra::Tensor<double,2> X({500,2});  // estado: [posición, velocidad]
    utec::algebra::Tensor<double,2> Y({500,1});  // acción: derecha=1, izquierda=0

    for (int i = 0; i < 500; i++) {
        double pos = (std::rand() % 200 - 100) / 100.0;
        double vel = (std::rand() % 200 - 100) / 100.0;

        X(i,0) = pos;
        X(i,1) = vel;

        // Política deseada: si posición+velocidad > 0 → derecha
        Y(i,0) = (pos + vel > 0) ? 1 : 0;
    }

    net.train<utec::neural_network::MSELoss>(X, Y, epochs, 32, 0.05);
}

void ControllerDemo::test() {
    utec::algebra::Tensor<double,2> t({1,2});
    t = {0.3, -0.1};

    auto pred = net.predict(t);

    std::cout << "\nControllerDemo: action = "
              << (pred(0,0) > 0.5 ? "RIGHT" : "LEFT")
              << " (score=" << pred(0,0) << ")\n";
}
