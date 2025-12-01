#pragma once
#include "nn_interfaces.h"
#include <numeric>
#include <functional>
#include <array>

namespace utec {
namespace neural_network {

template<typename T>
class Dense final : public ILayer<T> {
private:
    Tensor<T,2> W_;   // Pesos
    Tensor<T,2> b_;   // Bias
    Tensor<T,2> X_;   // Entrada
    Tensor<T,2> dW_;  // Gradiente de pesos
    Tensor<T,2> db_;  // Gradiente de bias

public:
    // Constructor con inicialización por funciones
    template<typename InitWFun, typename InitBFun>
    Dense(size_t in_f, size_t out_f, InitWFun init_w_fun, InitBFun init_b_fun) {
        W_ = Tensor<T,2>(std::array<size_t,2>{in_f, out_f});
        b_ = Tensor<T,2>(std::array<size_t,2>{1, out_f});
        init_w_fun(W_);
        init_b_fun(b_);
    }

    // Constructor rápido con constantes
    Dense(size_t in_f, size_t out_f, T init_w = static_cast<T>(0.1), T init_b = static_cast<T>(0.0)) {
        W_ = Tensor<T,2>(std::array<size_t,2>{in_f, out_f});
        b_ = Tensor<T,2>(std::array<size_t,2>{1, out_f});
        W_.fill(init_w);
        b_.fill(init_b);
    }

    Tensor<T,2> forward(const Tensor<T,2>& X) override {
        X_ = X;
        const auto rows = X.shape()[0];
        const auto out_features = W_.shape()[1];
        const auto in_features = W_.shape()[0];

        Tensor<T,2> Y(std::array<size_t,2>{rows, out_features});

        for(size_t i=0;i<rows;++i)
            for(size_t j=0;j<out_features;++j){
                T sum = static_cast<T>(0);
                for(size_t k=0;k<in_features;++k)
                    sum += X(i,k) * W_(k,j);
                Y(i,j) = sum + b_(0,j);
            }
        return Y;
    }

    Tensor<T,2> backward(const Tensor<T,2>& dZ) override {
        const size_t n_batches = X_.shape()[0];
        const size_t in_features = W_.shape()[0];
        const size_t out_features = W_.shape()[1];

        dW_ = Tensor<T,2>(std::array<size_t,2>{in_features, out_features});
        db_ = Tensor<T,2>(std::array<size_t,2>{1, out_features});

        // Gradientes de pesos
        for(size_t i=0;i<in_features;++i)
            for(size_t j=0;j<out_features;++j){
                T sum = 0;
                for(size_t k=0;k<n_batches;++k)
                    sum += X_(k,i) * dZ(k,j);
                dW_(i,j) = sum / static_cast<T>(n_batches);
            }

        // Gradientes de bias
        for(size_t j=0;j<out_features;++j){
            T sum = 0;
            for(size_t i=0;i<n_batches;++i)
                sum += dZ(i,j);
            db_(0,j) = sum / static_cast<T>(n_batches);
        }

        // Gradientes de entrada
        Tensor<T,2> dX(std::array<size_t,2>{n_batches, in_features});
        for(size_t i=0;i<n_batches;++i)
            for(size_t j=0;j<in_features;++j){
                T sum = 0;
                for(size_t k=0;k<out_features;++k)
                    sum += dZ(i,k) * W_(j,k);
                dX(i,j) = sum;
            }

        return dX;
    }

    void update_params(IOptimizer<T>& optimizer) override {
        optimizer.update(W_, dW_);
        optimizer.update(b_, db_);
    }

    const Tensor<T,2>& weights() const { return W_; }
    const Tensor<T,2>& bias() const { return b_; }
    const Tensor<T,2>& grad_weights() const { return dW_; }
    const Tensor<T,2>& grad_bias() const { return db_; }
};

} // namespace neural_network
} // namespace utec