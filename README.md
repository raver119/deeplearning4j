

<p align="center">
  <img src="logo.png">
</p>

 [![Documentation](https://img.shields.io/badge/api-reference-blue.svg)](https://deeplearning4j.org/docs/latest/)
[![Gitter](https://img.shields.io/gitter/room/deeplearning4j/deeplearning4j.svg)](https://gitter.im/deeplearning4j/deeplearning4j)

[Eclipse Deeplearning4j](https://www.deeplearning4j.org/)  is the first commercial-grade, open-source, distributed deep-learning library written for Java and Scala. Integrated with Hadoop and Apache Spark, DL4J brings AI to business environments for use on distributed GPUs and CPUs.


---

## Features

* DL4J takes advantage of the latest distributed computing frameworks including Apache Spark and Hadoop to accelerate training. On multi-GPUs, it is equal to Caffe in performance.

* The libraries are completely open-source, Apache 2.0, and maintained by the developer community and [Skymind](https://www.skymind.ai) team.

* Deeplearning4j is written in Java and is compatible with any JVM language, such as Scala, Clojure or Kotlin. The underlying computations are written in C, C++ and Cuda. Keras will serve as the Python API.


## Get started with Deeplearning4j


Hyperparameters are variables that determine how a neural network learns. They include how many times to update the weights of the model, how to initialize those weights, which activation function to attach to the nodes, which optimization algorithm to use, and how fast the model should learn. This is what one configuration would look like:

```java
    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
        .weightInit(WeightInit.XAVIER)
        .activation("relu")
        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
        .updater(new Sgd(0.05))
        // ... other hyperparameters
        .list()
        .backprop(true)
        .build();
```
With Deeplearning4j, you add a layer by calling layer on the NeuralNetConfiguration.Builder(), specifying its place in the order of layers (the zero-indexed layer below is the input layer), the number of input and output nodes, nIn and nOut, as well as the type: DenseLayer.

```java
.layer(0, new DenseLayer.Builder().nIn(784).nOut(250)
.build())
```

Once you have configured your net, you can train the model with `model.fit`.

## Keras Model Import

[Keras model import](https://github.com/eclipse/deeplearning4j/tree/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras) provides routines for importing neural network models originally configured and trained using Keras, a popular Python deep learning library.

Once you have imported your model into DL4J, our full production stack is at your disposal. We support import of all Keras model types, most layers and practically all utility functionality. Please check [here](http://deeplearning4j.org/docs/latest/keras-import-supported-features) for a complete list of supported Keras features.

Learn more about Keras model import here: [deeplearning4j.org/docs/latest/keras-import-overview](http://deeplearning4j.org/docs/latest/keras-import-overview)

## Install

* To install the latest release of Deeplearning4j, run examples and import Deeplearning4j in your projects, see our [Quickstart page](https://deeplearning4j.org/docs/latest/deeplearning4j-quickstart#Java).

* To build the Deeplearning4j stack from source, [this way](https://deeplearning4j.org/docs/latest/deeplearning4j-build-from-source).


## Examples

We have separate repository with various examples available: https://github.com/eclipse/deeplearning4j-examples

## Contribute

Checkout the contributor guidelines [here](https://github.com/eclipse/deeplearning4j/blob/master/CONTRIBUTING.md).


## Community Support

* Find a bug or requesting a new feature? Report all Eclipse Deeplearning4j issues on [Github](https://github.com/eclipse/deeplearning4j/issues).

* For help using DL4J, ND4J, DataVec, Arbiter, or any of our libraries [join our community chat on Gitter](https://gitter.im/deeplearning4j/deeplearning4j).

* We also monitor StackOverflow for any general usage questions about the DL4J libraries and Java.


## Resources

*   [2019-10-02 15:37:57 Wednesdayeeplearning4j.org](https://www.deeplearning4j.org)
*   [Deeplearning4j tutorials](https://deeplearning4j.org/tutorials/setup)
*   [Deeplearning4j examples](https://github.com/eclipse/deeplearning4j-examples)
*   [Book - Deep Learning: A Practitioner's Approach](https://www.amazon.com/Deep-Learning-Practitioners-Adam-Gibson/dp/1491914254)
*   [Deeplearning4j on Youtube](https://www.youtube.com/channel/UCa-HKBJwkfzs4AgZtdUuBXQ/videos)


## License

[Apache License 2.0](LICENSE)

<p align="center">
  <img src="https://avatars0.githubusercontent.com/u/8603402?s=280&v=4" width="50">
</p>
