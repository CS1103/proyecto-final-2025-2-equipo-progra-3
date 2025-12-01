#ifndef PONG_AI_TENSOR1_H
#define PONG_AI_TENSOR1_H

#pragma once
#include <iostream>
#include <array>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <type_traits>

namespace utec {
namespace algebra {

template <typename T, size_t N>
class Tensor {
    template <typename, size_t> friend class Tensor;

private:
    std::array<size_t, N> shape_;
    std::vector<T> data_;

public:
    Tensor() = default;

    template <typename... Args,
              typename = std::enable_if_t<(std::conjunction_v<std::is_arithmetic<Args>...>)>>
    Tensor(Args... dims) {
        if (sizeof...(dims) != N)
            throw std::invalid_argument("Number of dimensions do not match with " + std::to_string(N));
        size_t tmp[] = { static_cast<size_t>(dims)... };
        size_t total = 1;
        for (size_t i = 0; i < N; ++i) {
            shape_[i] = tmp[i];
            total *= tmp[i];
        }
        data_.resize(total);
    }

    void fill(const T& value) { std::fill(data_.begin(), data_.end(), value); }

    Tensor& operator=(std::initializer_list<T> list) {
        if (list.size() != data_.size())
            throw std::invalid_argument("Data size does not match tensor size");
        std::copy(list.begin(), list.end(), data_.begin());
        return *this;
    }

    const std::array<size_t, N>& shape() const { return shape_; }
    size_t size() const { return data_.size(); }

   
    size_t rows() const { 
        static_assert(N >= 1, "Tensor must have at least 1 dimension");
        return shape_[0]; 
    }
    
    size_t cols() const { 
        static_assert(N >= 2, "Tensor must have at least 2 dimensions for cols()");
        return shape_[1]; 
    }


    T& operator[](size_t i) { return data_[i]; }
    const T& operator[](size_t i) const { return data_[i]; }

    T& operator()(size_t i) { return data_[i]; }
    const T& operator()(size_t i) const { return data_[i]; }

    template <typename... Args>
    T& operator()(Args... args) {
        static_assert(sizeof...(args) == N, "Number of indices must match tensor rank");
        std::array<size_t, N> indices = { static_cast<size_t>(args)... };
        return data_[linear_index(indices)];
    }

    template <typename... Args>
    const T& operator()(Args... args) const {
        static_assert(sizeof...(args) == N, "Number of indices must match tensor rank");
        std::array<size_t, N> indices = { static_cast<size_t>(args)... };
        return data_[linear_index(indices)];
    }

    T& operator()(const std::array<size_t, N>& idx) { return data_[linear_index(idx)]; }
    const T& operator()(const std::array<size_t, N>& idx) const { return data_[linear_index(idx)]; }

    template <typename... Args>
    void reshape(Args... dims) {
        if (sizeof...(dims) != N)
            throw std::invalid_argument("Number of dimensions do not match with " + std::to_string(N));
        std::array<size_t, N> new_shape;
        size_t tmp[] = { static_cast<size_t>(dims)... };
        size_t new_total = 1;
        for (size_t i = 0; i < N; ++i) {
            new_shape[i] = tmp[i];
            new_total *= tmp[i];
        }
        if (new_total != data_.size()) data_.resize(new_total);
        shape_ = new_shape;
    }

    void reshape(const std::array<size_t, N>& dims) {
        size_t new_total = 1;
        for (size_t i = 0; i < N; ++i) {
            shape_[i] = dims[i];
            new_total *= dims[i];
        }
        if (new_total != data_.size()) data_.resize(new_total);
    }

    auto begin() { return data_.begin(); }
    auto end() { return data_.end(); }
    auto cbegin() const { return data_.cbegin(); }
    auto cend() const { return data_.cend(); }

    static std::array<size_t, N> broadcast_shape(const std::array<size_t, N>& a,
                                                const std::array<size_t, N>& b) {
        std::array<size_t, N> out{};
        for (int i = N - 1; i >= 0; --i) {
            if (a[i] == b[i]) out[i] = a[i];
            else if (a[i] == 1) out[i] = b[i];
            else if (b[i] == 1) out[i] = a[i];
            else throw std::invalid_argument("Shapes do not match and they are not compatible for broadcasting");
        }
        return out;
    }

    size_t linear_index(const std::array<size_t, N>& idx) const {
        size_t flat = 0, stride = 1;
        for (int dim = N - 1; dim >= 0; --dim) {
            flat += idx[dim] * stride;
            stride *= shape_[dim];
        }
        return flat;
    }

    template <typename Op>
    Tensor elementwise_broadcast(const Tensor& rhs, Op op) const {
        auto out_shape = broadcast_shape(shape_, rhs.shape_);
        Tensor result;
        result.shape_ = out_shape;

        size_t total = 1;
        for (auto s : out_shape) total *= s;
        result.data_.resize(total);

        std::array<size_t, N> idx{};
        for (size_t flat = 0; flat < total; ++flat) {
            size_t tmp = flat;
            for (int dim = N - 1; dim >= 0; --dim) {
                idx[dim] = tmp % out_shape[dim];
                tmp /= out_shape[dim];
            }

            std::array<size_t, N> idx_a{}, idx_b{};
            for (size_t i = 0; i < N; ++i) {
                idx_a[i] = (shape_[i] == 1) ? 0 : idx[i];
                idx_b[i] = (rhs.shape_[i] == 1) ? 0 : idx[i];
            }

            result.data_[flat] = op(
                data_[linear_index(idx_a)],
                rhs.data_[rhs.linear_index(idx_b)]
            );
        }
        return result;
    }

    Tensor operator+(const Tensor& rhs) const { return elementwise_broadcast(rhs, std::plus<T>()); }
    Tensor operator-(const Tensor& rhs) const { return elementwise_broadcast(rhs, std::minus<T>()); }
    Tensor operator*(const Tensor& rhs) const { return elementwise_broadcast(rhs, std::multiplies<T>()); }
    Tensor operator/(const Tensor& rhs) const { return elementwise_broadcast(rhs, std::divides<T>()); }

    Tensor operator*(const T& scalar) const {
        Tensor result = *this;
        for (auto& v : result.data_) v *= scalar;
        return result;
    }

    Tensor operator+(const T& scalar) const {
        Tensor result = *this;
        for (auto& v : result.data_) v += scalar;
        return result;
    }

    Tensor operator-(const T& scalar) const {
        Tensor result = *this;
        for (auto& v : result.data_) v -= scalar;
        return result;
    }

    Tensor operator/(const T& scalar) const {
        Tensor result = *this;
        for (auto& v : result.data_) v /= scalar;
        return result;
    }

    friend Tensor operator+(const T& scalar, const Tensor& t) { return t + scalar; }
    friend Tensor operator-(const T& scalar, const Tensor& t) {
        Tensor result = t;
        for (auto& v : result.data_) v = scalar - v;
        return result;
    }
    friend Tensor operator*(const T& scalar, const Tensor& t) { return t * scalar; }

    friend std::ostream& operator<<(std::ostream& os, const Tensor& t) {
        print_tensor(os, t, 0);
        return os;
    }

private:
    static void print_tensor(std::ostream& os, const Tensor& t, size_t depth) {
        if constexpr (N == 1) {
            for (size_t i = 0; i < t.shape_[0]; ++i) {
                os << t.data_[i];
                if (i + 1 < t.shape_[0]) os << " ";
            }
        } else {
            os << "{\n";
            size_t sub_size = 1;
            for (size_t i = 1; i < N; ++i) sub_size *= t.shape_[i];
            for (size_t i = 0; i < t.shape_[0]; ++i) {
                Tensor<T, N - 1> sub;
                fill_subtensor(sub, t, i * sub_size);
                for (size_t j = 0; j <= depth; ++j) os << "  ";
                Tensor<T, N - 1>::print_tensor(os, sub, depth + 1);
                os << "\n";
            }
            for (size_t j = 0; j < depth; ++j) os << "  ";
            os << "}";
        }
    }

    template <typename TT, size_t M>
    static void fill_subtensor(Tensor<TT, M>& sub, const Tensor<TT, M + 1>& parent, size_t start) {
        size_t copy_size = parent.data_.size() / parent.shape_[0];
        for (size_t i = 0; i < M; ++i) sub.shape_[i] = parent.shape_[i + 1];
        sub.data_.resize(copy_size);
        for (size_t i = 0; i < copy_size; ++i) sub.data_[i] = parent.data_[start + i];
    }
};


template <typename T, size_t N>
Tensor<T, N> transpose_2d(const Tensor<T, N>& t) {
    if constexpr (N == 1) {
        throw std::invalid_argument("Cannot transpose 1D tensor: need at least 2 dimensions");
    } else {
        auto s = t.shape();
        std::array<size_t, N> new_shape = s;

        std::swap(new_shape[N - 1], new_shape[N - 2]);


        Tensor<T, N> result;
        result.reshape(new_shape);

 
        std::array<size_t, N> idx{}, idx_out{};
        for (size_t flat = 0; flat < t.size(); ++flat) {

            size_t tmp = flat;
            for (int i = N - 1; i >= 0; --i) {
                idx[i] = tmp % s[i];
                tmp /= s[i];
            }

            idx_out = idx;
            std::swap(idx_out[N - 1], idx_out[N - 2]);

            result(idx_out) = t(idx);
        }
        return result;
    }
}


template <typename T, size_t N>
Tensor<T, N> matrix_product(const Tensor<T, N>& a, const Tensor<T, N>& b) {
    if constexpr (N < 2) throw std::invalid_argument("Need at least 2D tensors");

    auto s1 = a.shape();
    auto s2 = b.shape();

    size_t batch_dim = N > 2 ? s1[0] : 1;
    if constexpr (N > 2) {
        if (s1[0] != s2[0])
            throw std::invalid_argument("Matrix dimensions are compatible for multiplication BUT Batch dimensions do not match");
    }

    if (s1[N - 1] != s2[N - 2]) throw std::invalid_argument("Matrix dimensions are incompatible for multiplication");

    std::array<size_t, N> out_shape = s1;
    out_shape[N - 1] = s2[N - 1];
    Tensor<T, N> result;
    result.reshape(out_shape);

    std::array<size_t, N> idx_a{}, idx_b{}, idx_r{};
    for (size_t flat = 0; flat < result.size(); ++flat) {
        size_t tmp = flat;
        for (int i = N - 1; i >= 0; --i) {
            idx_r[i] = tmp % out_shape[i];
            tmp /= out_shape[i];
        }

        T sum = 0;
        for (size_t k = 0; k < s1[N - 1]; ++k) {
            idx_a = idx_r;
            idx_b = idx_r;
            idx_a[N - 1] = k;
            idx_b[N - 2] = k;
            sum += a(idx_a) * b(idx_b);
        }
        result(idx_r) = sum;
    }

    return result;
}

}  
}  

#endif 