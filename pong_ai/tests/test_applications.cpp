//
// Created by fabri on 1/12/2025.
//

#include <iostream>
#include "../include/apps/PatternClassifier.h"
#include "../include/apps/SequencePredictor.h"
#include "../include/utec/algebra/Tensor.h"

using namespace utec::algebra;

int main() {
    // ===== PatternClassifier =====
    PatternClassifier classifier(3);

    Tensor<double,2> X_class(4,3);
    X_class = {0.1,0.5,0.9,
               0.2,0.1,0.7,
               0.3,0.8,0.2,
               0.9,0.5,0.1};

    Tensor<double,2> Y_class(4,3);
    Y_class = {1,0,0,
               0,1,0,
               0,0,1,
               1,0,0};

    classifier.train(X_class, Y_class);

    auto output_class = classifier.predict(X_class);
    std::cout << "Predicciones PatternClassifier:" << std::endl << output_class << std::endl;

    // ===== SequencePredictor =====
    SequencePredictor predictor(4);

    Tensor<double,2> X_seq(5,4);
    X_seq = {1,2,3,4,
             2,3,4,5,
             3,4,5,6,
             4,5,6,7,
             5,6,7,8};

    Tensor<double,2> Y_seq(5,1);
    Y_seq = {5,6,7,8,9};

    predictor.train(X_seq, Y_seq);

    Tensor<double,2> last_seq(1,4);
    last_seq = {6,7,8,9};
    double next_value = predictor.predict_next(last_seq);

    std::cout << "Predicción siguiente valor SequencePredictor: " << next_value << std::endl;

    return 0;
}
