//
// Created by fabri on 30/11/2025.
//

#ifndef PONG_AI_NN_ACTIVATION_H
#define PONG_AI_NN_ACTIVATION_H

#pragma once
#include "nn_interfaces.h"
#include <cmath>
#include <algorithm>

namespace utec {
    namespace neural_network {

        template<typename T>
        class ReLU final : public ILayer<T> {
        private:
            Tensor<T,2> last_z_;
        public:
            Tensor<T,2> forward(const Tensor<T,2>& z) override {
                last_z_ = z;
                Tensor<T,2> result(z.shape()[0], z.shape()[1]);
                for (size_t i = 0; i < z.size(); ++i)
                    result[i] = std::max(static_cast<T>(0), z[i]);
                return result;
            }

            Tensor<T,2> backward(const Tensor<T,2>& g) override {
                Tensor<T,2> grad(g.shape()[0], g.shape()[1]);
                for (size_t i = 0; i < g.size(); ++i)
                    grad[i] = (last_z_[i] > static_cast<T>(0)) ? g[i] : static_cast<T>(0);
                return grad;
            }
        };

        template<typename T>
        class Sigmoid final : public ILayer<T> {
        private:
            Tensor<T,2> last_a_;
        public:
            Tensor<T,2> forward(const Tensor<T,2>& z) override {
                last_a_ = Tensor<T,2>(z.shape()[0], z.shape()[1]);
                for (size_t i = 0; i < z.size(); ++i) {
                    T val = z[i];
                    if (val < static_cast<T>(0)) {
                        T exp_val = std::exp(val);
                        last_a_[i] = exp_val / (static_cast<T>(1) + exp_val);
                    } else {
                        last_a_[i] = static_cast<T>(1) / (static_cast<T>(1) + std::exp(-val));
                    }
                }
                return last_a_;
            }

            Tensor<T,2> backward(const Tensor<T,2>& g) override {
                Tensor<T,2> grad(g.shape()[0], g.shape()[1]);
                for (size_t i = 0; i < g.size(); ++i) {
                    T a = last_a_[i];
                    grad[i] = g[i] * a * (static_cast<T>(1) - a);
                }
                return grad;
            }
        };

    }
}

#endif 