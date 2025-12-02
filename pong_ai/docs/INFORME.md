# Proyecto Final 2025-1: AI Neural Network
## **CS2013 Programación III** · Informe Final

### **Descripción**

>  Este proyecto implementa un framework de Deep Learning desarrollado desde cero en C++ Moderno (C++20). A diferencia de soluciones comerciales, este motor no utiliza librerías externas para el cálculo matricial; en su lugar, hemos implementado nuestro propio motor de álgebra lineal (Tensor<T,N>).


El sistema es capaz de resolver problemas de:
- Clasificación: Identificación de patrones (Dataset Iris). 
- Regresión: Predicción de secuencias numéricas. 
El desarrollo abarca los tres hitos (Epics) del curso:
- Epic 1: Biblioteca de Álgebra Tensorial. 
- Epic 2: Red Neuronal (Forward/Backward Propagation). 
- Epic 3: Aplicaciones Prácticas y Documentación.
### Contenidos

1. [Datos generales](#datos-generales)
2. [Requisitos e instalación](#requisitos-e-instalación)
3. [Investigación teórica](#1-investigación-teórica)
4. [Diseño e implementación](#2-diseño-e-implementación)
5. [Ejecución](#3-ejecución)
6. [Análisis del rendimiento](#4-análisis-del-rendimiento)
7. [Trabajo en equipo](#5-trabajo-en-equipo)
8. [Conclusiones](#6-conclusiones)
9. [Bibliografía](#7-bibliografía)
10. [Licencia](#licencia)
---

### Datos generales

* **Tema**: Redes Neuronales en AI
* **Grupo**: `equipo-progra-3`
* **Integrantes**:

  * Leonardo Gabriel Sanchez Terrazos – 202410273 (Responsable de investigación teórica)
  * Juan Diego Azabache – 2002110430 (Desarrollo de la arquitectura)
  * Pablo Mario Rodríguez Poémape – 202410047 (Implementación del modelo)
  * Fabrizio Gonzales Nuñez – 202110146 (Pruebas y benchmarking)
  * Eduardo Gabriel Osorio Panduro – 202410406 (Documentación y demo)


---

### Requisitos e instalación

1. **Compilador**: Compilador C++ compatible con C++20 (GCC 11+, Clang 12+, MSVC).
2. **Dependencias**:

   * CMake 3.18+
   * Eigen 3.4

4. **Instalación**:

   ```bash
   git clone https://github.com/CS1103/proyecto-final-2025-2-equipo-progra-3.git
   cd pong_ai
   mkdir build && cd build
   cmake ..
   make
   ```
---
# 1. Investigación Teórica

## 1.1 Historia y evolución de las Redes Neuronales

El estudio de las redes neuronales artificiales comenzó en 1943 con el modelo de McCulloch y Pitts, que intentaba representar matemáticamente el comportamiento de una neurona biológica. En 1957, Rosenblatt presentó el Perceptrón, capaz de aprender pesos mediante un algoritmo de ajuste, aunque limitado a problemas linealmente separables.

El avance más importante llegó en 1986, cuando Rumelhart, Hinton y Williams publicaron el algoritmo de Backpropagation, que permitió entrenar redes con múltiples capas e impulsó el desarrollo del aprendizaje profundo moderno. Durante los años 90 surgieron arquitecturas especializadas, como las redes convolucionales para visión y las recurrentes para secuencias.

Con la llegada de GPUs y grandes cantidades de datos, el aprendizaje profundo se consolidó como una de las áreas centrales de la inteligencia artificial. Este proyecto se inspira en esa evolución implementando desde cero el flujo completo de una red neuronal en C++.

---

## 1.2 Principales Arquitecturas

### 1.2.1 Perceptrón Multicapa (MLP)

El MLP está compuesto por capas densas donde cada neurona aplica una combinación lineal seguida de una activación no lineal:

$$
y = f(Wx + b)
$$

Aquí:
- $x$ es la entrada,
- $W$ es la matriz de pesos,
- $b$ el sesgo,
- $f$ una función como ReLU o Sigmoid.

El MLP es un modelo generalista adecuado para clasificación y regresión, y sirve como base para validar la implementación de forward y backward propagation dentro del framework del proyecto.

### 1.2.2 Redes Convolucionales (CNN)

Las CNN fueron diseñadas para datos espaciales como imágenes. Sus capas aplican filtros locales que recorren la entrada, permitiendo detectar patrones como bordes o texturas. Este mecanismo reduce la cantidad de parámetros y mejora la generalización.

Aunque no se implementan en este proyecto, el motor tensorial desarrollado permite futuras extensiones hacia este tipo de capas.

### 1.2.3 Redes Recurrentes (RNN)

Las RNN manejan datos secuenciales mediante conexiones que mantienen un estado interno, lo que les da memoria temporal. Variantes como LSTM y GRU permiten aprender dependencias de largo plazo sin sufrir del problema del gradiente que se desvanece.

El proyecto incluye un predictor de secuencias basado en MLP, pero su estructura modular permite evolucionar hacia modelos recurrentes.

---

## 1.3 Algoritmos de Entrenamiento

### 1.3.1 Forward Propagation

El proceso de inferencia consiste en propagar una entrada a través de cada capa aplicando transformaciones lineales y funciones de activación:

$$
z = Wx + b
$$

$$
a = f(z)
$$

Este flujo está implementado utilizando un motor tensorial propio en C++ que maneja operaciones básicas de álgebra lineal sin depender de bibliotecas externas.

---

### 1.3.2 Funciones de Costo

Las funciones de costo miden la discrepancia entre la predicción y el valor real. Entre las utilizadas en el proyecto están:

- Error Cuadrático Medio (MSE):
  $$
  L = \frac{1}{n} \sum (y_{\text{pred}} - y_{\text{true}})^2
  $$

- Entropía Cruzada Binaria (BCE)

Para clasificación multiclase, una mejora futura es implementar Softmax junto con Entropía Cruzada Categórica.

---

### 1.3.3 Backpropagation

El algoritmo de retropropagación calcula el gradiente de la función de costo respecto a cada parámetro de la red utilizando la regla de la cadena:

$$
\frac{\partial L}{\partial W} = 
\frac{\partial L}{\partial a}
\cdot
\frac{\partial a}{\partial z}
\cdot
\frac{\partial z}{\partial W}
$$

Cada capa del proyecto almacena sus gradientes y los utiliza para actualizar los pesos.

---

### 1.3.4 Optimizadores

Los optimizadores definen cómo se ajustan los parámetros durante el entrenamiento. Este proyecto implementa dos métodos fundamentales:
**SGD (Stochastic Gradient Descent)**  

$$
W_{\text{new}} = W_{\text{old}} - \eta \, \nabla W
$$


**Adam (Adaptive Moment Estimation)**  
Utiliza momentos acumulados de los gradientes y tasas de aprendizaje adaptativas, logrando una convergencia más rápida y estable.

Los dos algoritmos están implementados utilizando tensores propios, lo que permite un control completo sobre el cálculo.
---

### 2. Diseño e implementación

#### 2.1 Arquitectura de la solución

* **Estructura de carpetas**:

  ```
  pong_ai/
  ├── include/
  │   ├── utec/
  │   │   ├── algebra/       # EPIC 1: Motor Matemático
  │   │   └── nn/            # EPIC 2: Core de la Red Neuronal
  │   └── apps/              # EPIC 3: Casos de Uso (Wrappers)
  │       └── data/          # Datasets (iris.csv)
  ├── src/                   # Código fuente principal
  ├── tests/                 # Tests unitarios e integración
  └── docs/                  # Documentación
  ```
* **Descripción de archivos**:


  A. Motor de Álgebra (include/utec/algebra)
  - tensor.h: La clase base del proyecto. Implementa un array N-dimensional (Tensor<T, Rank>) usando templates variádicos. 
  - Funcionalidades: Broadcasting (suma de tensores de distinto tamaño), multiplicación de matrices optimizada, transpuesta y acceso multidimensional seguro.


  B. Core de Red Neuronal (include/utec/nn)
  - nn_interfaces.h: Define los contratos abstractos (ILayer, ILoss, IOptimizer). Permite polimorfismo, facilitando agregar nuevos tipos de capas sin tocar el código base. 
  - neural_network.h: La clase orquestadora. Mantiene un vector de capas (std::vector<unique_ptr<ILayer>>) y gestiona el bucle de entrenamiento (Epochs -> Forward -> Loss -> Backward -> Optimizer). 
  - nn_dense.h: Implementación de la capa totalmente conectada ($Y = XW + B$). Almacena los gradientes para el aprendizaje. 
  - nn_activation.h: Contiene ReLU (para capas ocultas) y Sigmoid (para salida probabilística). 
  - nn_optimizer.h: Implementa SGD (Descenso de Gradiente Estocástico) y Adam (Optimizador con momento adaptativo). 
  - nn_loss.h: Funciones de costo como MSELoss (Error Cuadrático Medio) para regresión y BCELoss (Binary Cross Entropy) para clasificación.


  C. Aplicaciones (include/apps)
  - PatternClassifier.h: Wrapper especializado para clasificación. Configura una topología de red específica (Entrada -> Dense -> ReLU -> Dense -> Sigmoid) para resolver el problema de las flores Iris. 
  - SequencePredictor.h: Wrapper para predicción de series numéricas. Demuestra la flexibilidad del framework para tareas de regresión.

#### 2.2 Manual de uso y casos de prueba

* **Cómo ejecutar**: `./pong_ai.exe`

* **Casos de prueba**:
Se incluyen ejecutables de prueba para verificar componentes aislados:

  - `./test_tensor.exe`: Valida operaciones matemáticas. 
  - `./test_neural_network.exe`: Prueba una red pequeña con datos dummy. 
  - `./test_applications.exe`: Valida SequencePredictor y PatternClassifier con datos sintéticos. 

### 3. Ejecución

> **Demo de ejemplo**: Video/demo alojado en `docs/demo.mp4`.
> Pasos:
>
> 1. Preparar datos de entrenamiento (formato CSV).
> 2. Ejecutar comando de entrenamiento.
> 3. Evaluar resultados con script de validación.



### 4. Análisis del rendimiento

Se realizaron pruebas exhaustivas utilizando el dataset Iris. A continuación, presentamos los dos escenarios principales de experimentación.

### **Escenario A: Clasificación Multiclase (3 Clases)**
    
* **Objetivo:** Clasificar simultáneamente Setosa, Versicolor y Virginica.
* **Configuración:** Red Neuronal Simple (4 inputs -> 3 outputs).
* **Métricas:**
     - Iteraciones: 1000 épocas
     - Tiempo total de entrenamiento: 6m20s
     - Exactitud (Accuracy): 64.6667%
     - Matriz de confusión:
```
         Predicted
Actual    0    1    2
    0    47    0    3   (Setosa: Bien clasificada)
    1     0    0   50   (Versicolor: Error total, confundida con Virginica)
    2     0    0   50   (Virginica: Clasificada como Virginica)
```

Análisis: La red actual, al ser un Perceptrón Multicapa simple, tiene dificultades para separar las clases Versicolor y Virginica debido a que sus características se superponen significativamente en el espacio vectorial. Se requiere una arquitectura más profunda o funciones de kernel para mejorar esto.

### **Escenario B: Clasificación Binaria (2 Clases)**

* **Objetivo:** Validar la corrección matemática del algoritmo. Se filtró el dataset para clasificar solo 2 clases (Setosa vs Virginica), normalizando los inputs entre 0 y 1.
* **Configuración:** Red Neuronal Simple (4 inputs -> 2 outputs).
* **Métricas:**
  - Iteraciones: 1000 épocas
  - Tiempo total de entrenamiento: 4m9s pero llega a una exactitud de 100% a los 3m5s con 700 épocas.
  - Exactitud (Accuracy): 100%
  - Matriz de confusión:

```
         Predicted
Actual    0    1
    0    50    0   (Setosa)
    1     0   50   (Virginica)
```

Análisis: El framework es robusto y funcional. La implementación matemática es correcta y capaz de aprender patrones linealmente separables a la perfección.


### 5. Trabajo en equipo

| Tarea                     | Miembro           | Rol                       |
| ------------------------- |-------------------| ------------------------- |
| Investigación teórica     | Leonardo Gabriel Sanchez Terrazos | Documentar bases teóricas |
| Diseño de la arquitectura | Juan Diego Azabache | UML y esquemas de clases  |
| Implementación del modelo | Pablo Mario Rodríguez Poémape | Código C++ de la NN       |
| Pruebas y benchmarking    | Fabrizio Gonzales | Generación de métricas    |
| Documentación y demo      | Fabrizio Gonzales | Tutorial y video demo     |


---

### 6. Conclusiones

#### **Conclusiones:**
- Independencia Tecnológica: Se logró implementar una red neuronal completa sin depender de frameworks externos (como PyTorch o TensorFlow), validando la comprensión profunda de los algoritmos de optimización y cálculo matricial.

- Correctitud Matemática: Los resultados en la clasificación binaria (100% de precisión) confirman que la implementación de los gradientes, la regla de la cadena en Backpropagation y la actualización de pesos funcionan matemáticamente como se espera.

- Modularidad: El diseño basado en interfaces (ILayer, IOptimizer) permite extender el proyecto fácilmente, añadiendo nuevas capas o métodos de optimización sin reescribir el núcleo.

#### **Mejoras Futuras:**
- Paralelización (SIMD/Multithreading): Actualmente las operaciones matriciales son secuenciales. Se propone implementar paralelismo utilizando hilos o instrucciones vectoriales (AVX) para acelerar el entrenamiento en datasets grandes.

- Función Softmax: Para mejorar el rendimiento en problemas multiclase (como el caso de 3 flores), es necesario implementar la función de activación Softmax en la capa de salida junto con Cross-Entropy Loss categórica.

- Serialización: Implementar métodos para guardar y cargar el modelo entrenado (pesos y sesgos) en archivos, permitiendo reutilizar la red sin necesidad de reentrenar cada vez.

---

### 7. Bibliografía

> [1] I. Goodfellow, Y. Bengio y A. Courville, Deep Learning. MIT Press, 2016.
> 
> [2] Y. LeCun et al., “Gradient-based learning applied to document recognition,” Proceedings of the IEEE, vol. 86, no. 11, pp. 2278–2324, 1998.
> 
> [3] D. E. Rumelhart, G. E. Hinton y R. J. Williams, “Learning representations by backpropagation,” Nature, vol. 323, pp. 533-536, 1986.
> 
> [4] R. Sutton y A. G. Barto, Reinforcement Learning: An Introduction, 2nd ed., MIT Press, 2018.
> 
> [5] X. Glorot y Y. Bengio, “Understanding the difficulty of training deep feedforward neural networks,” en AISTATS 2010, pp. 249–256.
> 
> [6] S. Haykin, Neural Networks and Learning Machines, 3rd ed., Pearson, 2009.
> 
> [7] G. James et al., An Introduction to Statistical Learning, Springer, 2013.
> 
> [8] R. A. Fisher, “The use of multiple measurements in taxonomic problems,” Annals of Eugenics, vol. 7, pp. 179–188, 1936.
---

### Licencia

Este proyecto usa la licencia **MIT**. Ver [LICENSE](LICENSE) para detalles.

---
