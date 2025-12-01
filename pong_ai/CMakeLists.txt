cmake_minimum_required(VERSION 4.0)
project(pong_ai)

set(CMAKE_CXX_STANDARD 20)

add_executable(pong_ai main.cpp
        include/apps/ControllerDemo.h
        include/apps/IApplicationModule.h
        include/apps/PatternClassifier.h
        include/apps/SequencePredictor.h

        include/utec/algebra/tensor.h

        include/utec/nn/nn_activation.h
        include/utec/nn/nn_dense.h
        include/utec/nn/nn_factories.h
        include/utec/nn/nn_interfaces.h
        include/utec/nn/nn_loss.h
        include/utec/nn/nn_optimizer.h
        include/utec/nn/neural_network.h
)

# === Tests ===
add_executable(test_tensor tests/test_tensor.cpp)
add_executable(test_neural_network tests/test_neural_network.cpp)
add_executable(test_applications tests/test_applications.cpp)
add_executable(prueba_3_classes tests/prueba_3_classes.cpp)
