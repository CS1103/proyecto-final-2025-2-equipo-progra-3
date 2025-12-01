//
// Created by fabri on 1/12/2025.
//

#include <iostream>
#include "../include/utec/algebra/tensor.h"

using namespace utec::algebra;

int main() {
    // Crear un tensor 2x3
    Tensor<double,2> A(2,3);
    A = {1,2,3,4,5,6};

    std::cout << "Tensor A:" << std::endl << A << std::endl;

    // Operaciones básicas
    Tensor<double,2> B(2,3);
    B.fill(10);

    std::cout << "Tensor B (filled with 10):" << std::endl << B << std::endl;

    return 0;
}
