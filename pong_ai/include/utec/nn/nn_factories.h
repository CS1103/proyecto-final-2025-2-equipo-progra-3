#pragma once
#include "nn_dense.h"
#include "nn_activation.h"
#include <memory>

namespace pong_ai {
    namespace nn_factory {

        using namespace utec::neural_network;

        struct LayerFactory {
            static std::unique_ptr<ILayer<double>> dense(size_t in, size_t out) {
                return std::make_unique<Dense<double>>(in, out, Init::xavier, Init::zeros);
            }

            static std::unique_ptr<ILayer<double>> relu() {
                return std::make_unique<ReLU<double>>();
            }

            static std::unique_ptr<ILayer<double>> sigmoid() {
                return std::make_unique<Sigmoid<double>>();
            }
        };

    } // namespace nn_factory
} // namespace pong_ai
