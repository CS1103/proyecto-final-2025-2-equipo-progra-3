//
// Created by fabri on 30/11/2025.
//

#ifndef PONG_AI_NN_INTERFACES_H
#define PONG_AI_NN_INTERFACES_H
#pragma once
#include "tensor.h"

namespace utec {
    namespace neural_network {

        template<typename T>
        class IOptimizer;

        using utec::algebra::Tensor;

        template<typename T>
        class ILayer {
        public:
            virtual ~ILayer() = default;
            virtual Tensor<T,2> forward(const Tensor<T,2>& z) = 0;
            virtual Tensor<T,2> backward(const Tensor<T,2>& g) = 0;
            virtual void update_params(IOptimizer<T>& optimizer) {}
        };

        template<typename T, size_t N>
        class ILoss {
        public:
            virtual ~ILoss() = default;
            virtual T loss() const = 0;
            virtual Tensor<T,N> loss_gradient() const = 0;
        };

        template<typename T>
        class IOptimizer {
        public:
            virtual ~IOptimizer() = default;
            virtual void update(Tensor<T, 2>& params, const Tensor<T, 2>& grads) = 0;
            virtual void step() {}
        };

    } // namespace neural_network
} // namespace utec

#endif //PONG_AI_NN_INTERFACES_H