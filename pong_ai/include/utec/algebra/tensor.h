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
    std::array<size_t, N> shape_{};
    std::vector<T> data_;

public:

    // -----------------------
    // DEFAULT CONSTRUCTOR
    // -----------------------
    Tensor() = default;

    // -----------------------
    // CONSTRUCTOR: dims...
    // -----------------------
    template <typename... Args,
              typename = std::enable_if_t<(std::conjunction_v<std::is_arithmetic<Args>...>)>>
    Tensor(Args... dims) {
        if (sizeof...(dims) != N)
            throw std::invalid_argument("Number of dimensions mismatch");

        size_t tmp[] = { static_cast<size_t>(dims)... };
        size_t total = 1;

        for (size_t i = 0; i < N; ++i) {
            shape_[i] = tmp[i];
            total *= tmp[i];
        }
        data_.resize(total);
    }

    // -----------------------
    // CONSTRUCTOR: from shape array
    // -----------------------
    Tensor(const std::array<size_t, N>& dims) {
        shape_ = dims;
        size_t total = 1;
        for (auto s : dims) total *= s;
        data_.assign(total, T());
    }

    // -----------------------
    // FILL
    // -----------------------
    void fill(const T& v) { std::fill(data_.begin(), data_.end(), v); }

    // -----------------------
    // ASSIGN FROM LIST
    // -----------------------
    Tensor& operator=(std::initializer_list<T> values) {
        if (values.size() != data_.size())
            throw std::invalid_argument("Initializer list does not match tensor size");

        std::copy(values.begin(), values.end(), data_.begin());
        return *this;
    }

    // -----------------------
    // GETTERS
    // -----------------------
    const std::array<size_t, N>& shape() const { return shape_; }
    size_t size() const { return data_.size(); }

    // -----------------------
    // OPERATOR []
    // -----------------------
    T& operator[](size_t i) { return data_[i]; }
    const T& operator[](size_t i) const { return data_[i]; }

    // -----------------------
    // 1D ACCESS
    // -----------------------
    T& operator()(size_t i) { return data_[i]; }
    const T& operator()(size_t i) const { return data_[i]; }

    // -----------------------
    // MULTI-INDEX ACCESS
    // -----------------------
    template <typename... Args>
    T& operator()(Args... args) {
        static_assert(sizeof...(args) == N);
        std::array<size_t, N> idx{ static_cast<size_t>(args)... };
        return data_[linear_index(idx)];
    }

    template <typename... Args>
    const T& operator()(Args... args) const {
        static_assert(sizeof...(args) == N);
        std::array<size_t, N> idx{ static_cast<size_t>(args)... };
        return data_[linear_index(idx)];
    }

    T& operator()(const std::array<size_t, N>& idx) { return data_[linear_index(idx)]; }
    const T& operator()(const std::array<size_t, N>& idx) const { return data_[linear_index(idx)]; }

    // -----------------------
    // RESHAPE SAFE
    // -----------------------
    void reshape(const std::array<size_t, N>& dims) {
        size_t total_new = 1;
        for (auto d : dims) total_new *= d;

        if (total_new != data_.size())
            throw std::invalid_argument("reshape cannot change total size");

        shape_ = dims;
    }

    template <typename... Args>
    void reshape(Args... dims) {
        static_assert(sizeof...(dims) == N);

        std::array<size_t, N> arr{ static_cast<size_t>(dims)... };
        reshape(arr);
    }

    // -----------------------
    // ITERATORS
    // -----------------------
    auto begin() { return data_.begin(); }
    auto end() { return data_.end(); }
    auto cbegin() const { return data_.cbegin(); }
    auto cend() const { return data_.cend(); }

    // -----------------------
    // LINEAR INDEX
    // -----------------------
    size_t linear_index(const std::array<size_t, N>& idx) const {
        size_t flat = 0, stride = 1;
        for (int i = N - 1; i >= 0; --i) {
            flat += idx[i] * stride;
            stride *= shape_[i];
        }
        return flat;
    }

    // -----------------------
    // BROADCAST SHAPE
    // -----------------------
    static std::array<size_t, N> broadcast_shape(const std::array<size_t, N>& a,
                                                 const std::array<size_t, N>& b) {
        std::array<size_t, N> out{};
        for (int i = N - 1; i >= 0; --i) {
            if (a[i] == b[i]) out[i] = a[i];
            else if (a[i] == 1) out[i] = b[i];
            else if (b[i] == 1) out[i] = a[i];
            else throw std::invalid_argument("Broadcast error");
        }
        return out;
    }

    // -----------------------
    // ELEMENTWISE OP
    // -----------------------
    template <typename Op>
    Tensor elementwise_broadcast(const Tensor& rhs, Op op) const {
        auto out_shape = broadcast_shape(shape_, rhs.shape_);
        Tensor result(out_shape);

        size_t total = result.size();
        std::array<size_t, N> idx{}, idx_a{}, idx_b{};

        for (size_t flat = 0; flat < total; ++flat) {
            size_t tmp = flat;

            for (int dim = N - 1; dim >= 0; --dim) {
                idx[dim] = tmp % out_shape[dim];
                tmp /= out_shape[dim];
            }

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

    // -----------------------
    // BASIC OPERATORS
    // -----------------------
    Tensor operator+(const Tensor& t) const { return elementwise_broadcast(t, std::plus<T>()); }
    Tensor operator-(const Tensor& t) const { return elementwise_broadcast(t, std::minus<T>()); }
    Tensor operator*(const Tensor& t) const { return elementwise_broadcast(t, std::multiplies<T>()); }
    Tensor operator/(const Tensor& t) const { return elementwise_broadcast(t, std::divides<T>()); }

    Tensor operator*(T s) const {
        Tensor r = *this;
        for (auto& v : r.data_) v *= s;
        return r;
    }

    // -----------------------
    // PRINT
    // -----------------------
    friend std::ostream& operator<<(std::ostream& os, const Tensor& t) {
        t.print(os, 0);
        return os;
    }

private:

    void print(std::ostream& os, size_t depth) const {
        if constexpr (N == 1) {
            for (size_t i = 0; i < shape_[0]; ++i) {
                os << data_[i] << (i + 1 < shape_[0] ? " " : "");
            }
        } else {
            os << "{\n";
            size_t inner = 1;
            for (size_t i = 1; i < N; ++i) inner *= shape_[i];

            for (size_t i = 0; i < shape_[0]; ++i) {
                for (size_t d = 0; d <= depth; ++d) os << "  ";

                Tensor<T, N - 1> view;
                view.shape_.fill(0);

                for (size_t k = 0; k < N - 1; ++k)
                    view.shape_[k] = shape_[k + 1];

                view.data_.resize(inner);

                for (size_t k = 0; k < inner; ++k)
                    view.data_[k] = data_[i * inner + k];

                view.print(os, depth + 1);
                os << "\n";
            }

            for (size_t d = 0; d < depth; ++d) os << "  ";
            os << "}";
        }
    }
};

// -----------------------------------------------------
// TRANSPOSE 2D
// -----------------------------------------------------
template <typename T, size_t N>
Tensor<T, N> transpose_2d(const Tensor<T, N>& t) {
    if constexpr (N < 2)
        throw std::invalid_argument("Cannot transpose 1D tensor");

    auto s = t.shape();
    std::array<size_t, N> new_shape = s;
    std::swap(new_shape[N - 1], new_shape[N - 2]);

    Tensor<T, N> result(new_shape);

    std::array<size_t, N> idx{}, idx_out{};
    for (size_t flat = 0; flat < t.size(); ++flat) {
        size_t tmp = flat;
        for (int i = N - 1; i >= 0; --i) {
            idx[i] = tmp % s[i];
            tmp /= s[i];
        }

        idx_out = idx;
        std::swap(idx_out[N - 2], idx_out[N - 1]);

        result(idx_out) = t(idx);
    }
    return result;
}

// -----------------------------------------------------
// MATRIX PRODUCT
// -----------------------------------------------------
template <typename T, size_t N>
Tensor<T, N> matrix_product(const Tensor<T, N>& a, const Tensor<T, N>& b) {
    if constexpr (N < 2)
        throw std::invalid_argument("Need at least 2D");

    auto s1 = a.shape();
    auto s2 = b.shape();

    if (s1[N - 1] != s2[N - 2])
        throw std::invalid_argument("Incompatible shapes for matmul");

    std::array<size_t, N> out_shape = s1;
    out_shape[N - 1] = s2[N - 1];

    Tensor<T, N> result(out_shape);

    std::array<size_t, N> idx{}, idxA{}, idxB{};

    for (size_t flat = 0; flat < result.size(); ++flat) {
        size_t tmp = flat;
        for (int i = N - 1; i >= 0; --i) {
            idx[i] = tmp % out_shape[i];
            tmp /= out_shape[i];
        }

        T sum = 0;
        for (size_t k = 0; k < s1[N - 1]; ++k) {
            idxA = idx;
            idxB = idx;
            idxA[N - 1] = k;
            idxB[N - 2] = k;
            sum += a(idxA) * b(idxB);
        }

        result(idx) = sum;
    }

    return result;
}

}  // namespace algebra
}  // namespace utec
