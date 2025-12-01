#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <algorithm>

#include "../include/apps/PatternClassifier.h"
#include "../include/utec/algebra/tensor.h"

using namespace utec::algebra;

int main() {
    // Mapa de etiquetas
    std::map<std::string,int> label_map{
        {"Setosa",0},
        {"Versicolor",1},
        {"Virginica",2}
    };

    // Cargar iris.csv
    std::ifstream file("../include/apps/data/iris.csv");
    if(!file.is_open()) {
        std::cerr << "No se pudo abrir iris.csv" << std::endl;
        return 1;
    }

    std::vector<std::vector<double>> X_values;
    std::vector<std::vector<double>> Y_values;

    std::string line;
    bool header_skipped = false;
    while(std::getline(file, line)) {
        if(!header_skipped) { header_skipped = true; continue; } // Saltar header
        if(line.empty()) continue;
        std::stringstream ss(line);
        std::string item;
        std::vector<double> features;
        for(int i=0;i<4;i++) {
            std::getline(ss, item, ',');
            features.push_back(std::stod(item));
        }
        std::getline(ss, item, ','); // etiqueta
        item.erase(std::remove(item.begin(), item.end(), '"'), item.end()); // quitar comillas

        // --- Todas las clases ---
        X_values.push_back(features);
        std::vector<double> label_onehot(3,0.0);
        label_onehot[label_map[item]] = 1.0;
        Y_values.push_back(label_onehot);
    }

    size_t n_samples = X_values.size();
    size_t n_features = X_values[0].size();

    Tensor<double,2> X(n_samples, n_features);
    Tensor<double,2> Y(n_samples, 3); // Tres clases

    // Copiar datos
    for(size_t i=0;i<n_samples;++i){
        for(size_t j=0;j<n_features;++j)
            X(i,j) = X_values[i][j];
        for(size_t j=0;j<3;++j)
            Y(i,j) = Y_values[i][j];
    }

    // ===== Normalizar features entre 0 y 1 =====
    for(size_t j=0;j<n_features;++j){
        double min_val = X(0,j), max_val = X(0,j);
        for(size_t i=0;i<n_samples;++i){
            if(X(i,j) < min_val) min_val = X(i,j);
            if(X(i,j) > max_val) max_val = X(i,j);
        }
        for(size_t i=0;i<n_samples;++i){
            X(i,j) = (X(i,j) - min_val) / (max_val - min_val);
        }
    }

    // ===== Clasificador =====
    PatternClassifier classifier(n_features);

    int epochs = 1000; // puedes ajustar
    for(int e=0;e<epochs;++e){
        classifier.train(X,Y);

        if((e+1) % 50 == 0){ // imprimir cada 50 epochs
            auto output = classifier.predict(X);
            int correct = 0;
            for(size_t i=0;i<n_samples;++i){
                int predicted = std::distance(&output(i,0), std::max_element(&output(i,0), &output(i,0)+3));
                int actual = std::distance(&Y(i,0), std::max_element(&Y(i,0), &Y(i,0)+3));
                if(predicted == actual) correct++;
            }
            double accuracy = 100.0 * correct / n_samples;
            std::cout << "Epoch " << e+1 << " - Accuracy: " << accuracy << "%\n";
        }
    }

    // ===== Predicciones finales =====
    auto output = classifier.predict(X);
    std::cout << "\nPredicciones PatternClassifier (3 clases):\n" << output << std::endl;

    // ===== Exactitud y matriz de confusión =====
    int correct = 0;
    int confusion[3][3] = {0};

    for(size_t i=0;i<n_samples;++i){
        int predicted = std::distance(&output(i,0), std::max_element(&output(i,0), &output(i,0)+3));
        int actual = std::distance(&Y(i,0), std::max_element(&Y(i,0), &Y(i,0)+3));
        if(predicted == actual) correct++;
        confusion[actual][predicted]++;
    }

    double accuracy = 100.0 * correct / n_samples;
    std::cout << "\nExactitud final (3 clases): " << accuracy << "%" << std::endl;

    std::cout << "\nMatriz de confusión:\n";
    std::cout << "           Predicted\n";
    std::cout << "Actual    0    1    2\n";
    for(int i=0;i<3;++i){
        std::cout << "    " << i;
        for(int j=0;j<3;++j){
            std::cout << "    " << confusion[i][j];
        }
        std::cout << "\n";
    }

    return 0;
}
