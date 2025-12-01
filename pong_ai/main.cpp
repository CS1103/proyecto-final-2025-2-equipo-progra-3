#include "include/utec/apps/PatternClassifier.h"
#include "include/utec/apps/ControllerDemo.h"
#include "include/utec/apps/SequencePredictor.h"
#include <iostream>

int main() {
    std::cout << "=== Neural Network Applications Demo ===\n\n";

    std::cout << "1. PATTERN CLASSIFIER (Circle, Square, Triangle)\n";
    std::cout << "------------------------------------------------\n";
    PatternClassifier classifier;
    std::cout << "Training pattern classifier...\n";
    classifier.train(2000);
    std::cout << "Testing pattern classifier:\n";
    classifier.test();
    std::cout << "\n";

    std::cout << "2. SEQUENCE PREDICTOR (Predict next number)\n";
    std::cout << "--------------------------------------------\n";
    SequencePredictor predictor;
    std::cout << "Training sequence predictor...\n";
    predictor.train(3000);
    std::cout << "Testing sequence predictor:";
    predictor.test();
    std::cout << "\n";

    std::cout << "3. CONTROLLER DEMO (Left/Right decision)\n";
    std::cout << "----------------------------------------\n";
    ControllerDemo controller;
    std::cout << "Training controller...\n";
    controller.train(2500);
    std::cout << "Testing controller:";
    controller.test();
    std::cout << "\n";

    std::cout << "=== All demos completed successfully! ===\n";
    return 0;
}
