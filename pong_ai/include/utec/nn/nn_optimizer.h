//
// Created by fabri on 30/11/2025.
//

#ifndef PONG_AI_NN_OPTIMIZER_H
#define PONG_AI_NN_OPTIMIZER_H
#pragma once
#include "nn_interfaces.h"
#include <cmath>
#include <vector>

namespace utec {
    namespace neural_network {

        template<typename T>
        class SGD final : public IOptimizer<T> {
        private:
            T lr_;
        public:
            explicit SGD(T learning_rate = 0.01) : lr_(learning_rate) {}
            void update(Tensor<T, 2>& params, const Tensor<T, 2>& grads) override {
                for (size_t i = 0; i < params.size(); ++i)
                    params[i] -= lr_ * grads[i];
            }
        };

        template<typename T>
        class Adam final : public IOptimizer<T> {
        private:
            T lr_, beta1_, beta2_, eps_;
            size_t t_;
            std::vector<Tensor<T,2>> m_, v_;
        public:
            explicit Adam(T learning_rate = 0.001, T beta1 = 0.9, T beta2 = 0.999, T epsilon = 1e-8)
                : lr_(learning_rate), beta1_(beta1), beta2_(beta2), eps_(epsilon), t_(0) {}

            void update(Tensor<T, 2>& params, const Tensor<T, 2>& grads) override {
                if (m_.size() < 1) {
                    m_.emplace_back(params.shape());
                    v_.emplace_back(params.shape());
                    m_[0].fill(0);
                    v_[0].fill(0);
                }
                auto& m = m_[0];
                auto& v = v_[0];
                ++t_;
                for (size_t i = 0; i < params.size(); ++i) {
                    m[i] = beta1_ * m[i] + (1 - beta1_) * grads[i];
                    v[i] = beta2_ * v[i] + (1 - beta2_) * grads[i] * grads[i];
                    T m_hat = m[i] / (1 - std::pow(beta1_, t_));
                    T v_hat = v[i] / (1 - std::pow(beta2_, t_));
                    params[i] -= lr_ * m_hat / (std::sqrt(v_hat) + eps_);
                }
            }
            void step() override { ++t_; }
        };

    }
}

#endif //PONG_AI_NN_OPTIMIZER_H