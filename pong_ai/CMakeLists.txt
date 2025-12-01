cmake_minimum_required(VERSION 4.0)
project(pong_ai)

set(CMAKE_CXX_STANDARD 20)

add_executable(pong_ai main.cpp
        src/utec/apps/PatternClassifier.cpp
        include/utec/apps/PatternClassifier.h
        src/utec/apps/SequencePredictor.cpp
        include/utec/apps/SequencePredictor.h
        src/utec/apps/ControllerDemo.cpp
        include/utec/apps/ControllerDemo.h
        include/utec/algebra/tensor.h
        include/utec/nn/nn_dense.h
        include/utec/nn/neural_network.h
        include/utec/nn/nn_activation.h
        include/utec/nn/nn_interfaces.h
        include/utec/nn/nn_loss.h
        include/utec/nn/nn_optimizer.h
)
