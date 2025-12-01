//
// Created by fabri on 30/11/2025.
//

#ifndef PONG_AI_NEURAL_NETWORK_H
#define PONG_AI_NEURAL_NETWORK_H

#pragma once
#include "nn_interfaces.h"
#include "nn_loss.h"
#include "nn_optimizer.h"
#include <memory>
#include <vector>

namespace utec {
    namespace neural_network {

        template<typename T>
        class NeuralNetwork {
        private:
            std::vector<std::unique_ptr<ILayer<T>>> layers_;

        public:
            void add_layer(std::unique_ptr<ILayer<T>> layer) {
                layers_.push_back(std::move(layer));
            }

            Tensor<T,2> predict(const Tensor<T,2>& X) {
                Tensor<T,2> output = X;
                for (auto& layer : layers_)
                    output = layer->forward(output);
                return output;
            }

            template <
                template <typename ...> class LossType,
                template <typename ...> class OptimizerType = SGD
            >
            void train(const Tensor<T,2>& X, const Tensor<T,2>& Y,
                       const size_t epochs, const size_t batch_size, T learning_rate)
            {
                OptimizerType<T> optimizer(learning_rate);

                for (size_t epoch = 0; epoch < epochs; ++epoch) {
                    Tensor<T,2> output = X;

                    // Forward pass
                    for (auto& layer : layers_)
                        output = layer->forward(output);

                    // Calcular pérdida
                    LossType<T> loss_fn(output, Y);

                    // Backward pass
                    Tensor<T,2> grad = loss_fn.loss_gradient();
                    for (int i = static_cast<int>(layers_.size()) - 1; i >= 0; --i)
                        grad = layers_[i]->backward(grad);

                    // Actualizar parámetros
                    for (auto& layer : layers_)
                        layer->update_params(optimizer);
                }
            }
        };

    } // namespace neural_network
} // namespace utec

#endif //PONG_AI_NEURAL_NETWORK_H