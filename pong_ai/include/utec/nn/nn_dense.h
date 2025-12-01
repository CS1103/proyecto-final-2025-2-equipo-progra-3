//
// Created by fabri on 30/11/2025.
//

#ifndef PONG_AI_NN_DENSE_H
#define PONG_AI_NN_DENSE_H

#pragma once
#include "nn_interfaces.h"
#include <numeric>

namespace utec {
namespace neural_network {

template<typename T>
class Dense final : public ILayer<T> {
private:
    Tensor<T,2> W_;   
    Tensor<T,2> b_;   
    Tensor<T,2> X_;   
    Tensor<T,2> dW_;  
    Tensor<T,2> db_;  

public:
    template<typename InitWFun, typename InitBFun>
    Dense(size_t in_f, size_t out_f, InitWFun init_w_fun, InitBFun init_b_fun) {
        W_ = Tensor<T,2>(in_f, out_f);
        b_ = Tensor<T,2>(1, out_f);
        init_w_fun(W_);
        init_b_fun(b_);
    }

    Tensor<T,2> forward(const Tensor<T,2>& X) override {
        X_ = X;
        const auto rows = X.shape()[0];
        const auto out_features = W_.shape()[1];
        Tensor<T,2> Y(rows, out_features);

        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < out_features; ++j) {
                T sum = 0;
                for (size_t k = 0; k < W_.shape()[0]; ++k)
                    sum += X(i, k) * W_(k, j);
                Y(i, j) = sum + b_(0, j);
            }
        }
        return Y;
    }

    Tensor<T,2> backward(const Tensor<T,2>& dZ) override {
        const size_t n_batches = X_.shape()[0];
        const size_t in_features = W_.shape()[0];
        const size_t out_features = W_.shape()[1];

        dW_ = Tensor<T,2>(in_features, out_features);
        for (size_t i = 0; i < in_features; ++i) {
            for (size_t j = 0; j < out_features; ++j) {
                T sum = 0;
                for (size_t k = 0; k < n_batches; ++k)
                    sum += X_(k, i) * dZ(k, j);
                dW_(i, j) = sum / static_cast<T>(n_batches);
            }
        }

        db_ = Tensor<T,2>(1, out_features);
        for (size_t j = 0; j < out_features; ++j) {
            T sum = 0;
            for (size_t i = 0; i < n_batches; ++i)
                sum += dZ(i, j);
            db_(0, j) = sum / static_cast<T>(n_batches);
        }

        Tensor<T,2> dX(n_batches, in_features);
        for (size_t i = 0; i < n_batches; ++i) {
            for (size_t j = 0; j < in_features; ++j) {
                T sum = 0;
                for (size_t k = 0; k < out_features; ++k)
                    sum += dZ(i, k) * W_(j, k);
                dX(i, j) = sum;
            }
        }

        return dX;
    }

    void update_params(IOptimizer<T>& optimizer) override {
        optimizer.update(W_, dW_);
        optimizer.update(b_, db_);
    }
};

}
}

#endif 