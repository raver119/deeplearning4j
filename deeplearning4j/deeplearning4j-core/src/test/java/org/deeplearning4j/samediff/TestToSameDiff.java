/*
 * ******************************************************************************
 *  * Copyright (c) 2020 Konduit K.K.
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.samediff;

import static org.junit.Assert.*;

import com.google.common.collect.Lists;
import com.google.common.collect.MapMaker;
import com.google.common.collect.Maps;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.math3.ml.neuralnet.MapUtils;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration.ListBuilder;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.LossLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.updater.BaseMultiLayerUpdater;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.util.ToSameDiffUtils;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import org.nd4j.autodiff.listeners.records.History;
import org.nd4j.autodiff.loss.LossReduce;
import org.nd4j.autodiff.samediff.SDIndex;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.internal.memory.NoOpMemoryMgr;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.impl.ActivationSigmoid;
import org.nd4j.linalg.activations.impl.ActivationSoftmax;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.adapter.SingletonDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.TestDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.config.NoOp;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.learning.regularization.L1Regularization;
import org.nd4j.linalg.learning.regularization.L2Regularization;
import org.nd4j.linalg.learning.regularization.Regularization;
import org.nd4j.linalg.learning.regularization.WeightDecay;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.nd4j.linalg.lossfunctions.impl.LossBinaryXENT;
import org.nd4j.linalg.lossfunctions.impl.LossCosineProximity;
import org.nd4j.linalg.lossfunctions.impl.LossL1;
import org.nd4j.linalg.lossfunctions.impl.LossMSE;

@Slf4j
public class TestToSameDiff extends BaseDL4JTest {

    private static final String expectedSummary = "--- Summary ---\n"
            + "Variables:               30                   (8 with arrays)\n"
            + "Functions:               20                  \n"
            + "SameDiff Function Defs:  0                   \n"
            + "Loss function variables: [loss]\n"
            + "\n"
            + "--- Variables ---\n"
            + "- Name -                          - Array Shape -     - Variable Type -   - Data Type-        - Output Of Function -                     - Inputs To Functions -\n"
            + "input                             [-1, 1, 28, 28]     PLACEHOLDER         FLOAT               <none>                                     [layer0/inputPreprocessor/reshape]\n"
            + "layer0/inputPreprocessor/reshape  -                   ARRAY               FLOAT               layer0/inputPreprocessor/reshape(reshape)  [layer0/conv2d]     \n"
            + "layer0/b                          [1, 20]             VARIABLE            FLOAT               <none>                                     [layer0/conv2d]     \n"
            + "layer0/W                          [20, 1, 5, 5]       VARIABLE            FLOAT               <none>                                     [layer0/conv2d]     \n"
            + "layer0/conv2d                     -                   ARRAY               FLOAT               layer0/conv2d(conv2d)                      [layer1/maxpool2d]  \n"
            + "layer1/maxpool2d                  -                   ARRAY               FLOAT               layer1/maxpool2d(maxpool2d)                [layer2/conv2d]     \n"
            + "layer2/b                          [1, 50]             VARIABLE            FLOAT               <none>                                     [layer2/conv2d]     \n"
            + "layer2/W                          [50, 20, 5, 5]      VARIABLE            FLOAT               <none>                                     [layer2/conv2d]     \n"
            + "layer2/conv2d                     -                   ARRAY               FLOAT               layer2/conv2d(conv2d)                      [layer3/maxpool2d]  \n"
            + "layer3/maxpool2d                  -                   ARRAY               FLOAT               layer3/maxpool2d(maxpool2d)                [layer4/inputPreprocessor/reshape]\n"
            + "layer4/inputPreprocessor/reshape  -                   ARRAY               FLOAT               layer4/inputPreprocessor/reshape(reshape)  [layer4/mmul]       \n"
            + "layer4/b                          [1, 500]            VARIABLE            FLOAT               <none>                                     [layer4/add]        \n"
            + "layer4/W                          [800, 500]          VARIABLE            FLOAT               <none>                                     [layer4/mmul]       \n"
            + "layer4/mmul                       -                   ARRAY               FLOAT               layer4/mmul(mmul)                          [layer4/add]        \n"
            + "layer4/add                        -                   ARRAY               FLOAT               layer4/add(add)                            [layer4/relu]       \n"
            + "layer4/relu                       -                   ARRAY               FLOAT               layer4/relu(relu)                          [layer5/mmul]       \n"
            + "layer5/b                          [1, 10]             VARIABLE            FLOAT               <none>                                     [layer5/add]        \n"
            + "layer5/W                          [500, 10]           VARIABLE            FLOAT               <none>                                     [layer5/mmul]       \n"
            + "layer5/mmul                       -                   ARRAY               FLOAT               layer5/mmul(mmul)                          [layer5/add]        \n"
            + "layer5/add                        -                   ARRAY               FLOAT               layer5/add(add)                            [layer5/softmax]    \n"
            + "layer5/softmax                    -                   ARRAY               FLOAT               layer5/softmax(softmax)                    [layer5/loss/ClipByValue]\n"
            + "labels                            [-1, 10]            PLACEHOLDER         FLOAT               <none>                                     [layer5/loss/multiply, layer5/loss/size_at]\n"
            + "layer5/loss/ClipByValue           -                   ARRAY               FLOAT               layer5/loss/ClipByValue(ClipByValue)       [layer5/loss/log]   \n"
            + "layer5/loss/log                   -                   ARRAY               FLOAT               layer5/loss/log(log)                       [layer5/loss/multiply]\n"
            + "layer5/loss/multiply              -                   ARRAY               FLOAT               layer5/loss/multiply(multiply)             [layer5/loss/neg]   \n"
            + "layer5/loss/neg                   -                   ARRAY               FLOAT               layer5/loss/neg(neg)                       [layer5/loss/reduce_sum]\n"
            + "layer5/loss/reduce_sum            -                   ARRAY               FLOAT               layer5/loss/reduce_sum(reduce_sum)         [layer5/loss/divide]\n"
            + "layer5/loss/size_at               -                   ARRAY               LONG                layer5/loss/size_at(size_at)               [layer5/loss/cast]  \n"
            + "layer5/loss/cast                  -                   ARRAY               FLOAT               layer5/loss/cast(cast)                     [layer5/loss/divide]\n"
            + "loss                              -                   ARRAY               FLOAT               layer5/loss/divide(divide)                                     \n"
            + "\n"
            + "\n"
            + "--- Functions ---\n"
            + "     - Function Name -                 - Op -           - Inputs -                                              - Outputs -                         \n"
            + "0    layer0/inputPreprocessor/reshape  Reshape          [input]                                                 [layer0/inputPreprocessor/reshape]  \n"
            + "1    layer0/conv2d                     Conv2D           [layer0/inputPreprocessor/reshape, layer0/W, layer0/b]  [layer0/conv2d]                     \n"
            + "2    layer1/maxpool2d                  MaxPooling2D     [layer0/conv2d]                                         [layer1/maxpool2d]                  \n"
            + "3    layer2/conv2d                     Conv2D           [layer1/maxpool2d, layer2/W, layer2/b]                  [layer2/conv2d]                     \n"
            + "4    layer3/maxpool2d                  MaxPooling2D     [layer2/conv2d]                                         [layer3/maxpool2d]                  \n"
            + "5    layer4/inputPreprocessor/reshape  Reshape          [layer3/maxpool2d]                                      [layer4/inputPreprocessor/reshape]  \n"
            + "6    layer4/mmul                       Mmul             [layer4/inputPreprocessor/reshape, layer4/W]            [layer4/mmul]                       \n"
            + "7    layer4/add                        AddOp            [layer4/mmul, layer4/b]                                 [layer4/add]                        \n"
            + "8    layer4/relu                       RectifiedLinear  [layer4/add]                                            [layer4/relu]                       \n"
            + "9    layer5/mmul                       Mmul             [layer4/relu, layer5/W]                                 [layer5/mmul]                       \n"
            + "10   layer5/add                        AddOp            [layer5/mmul, layer5/b]                                 [layer5/add]                        \n"
            + "11   layer5/softmax                    SoftMax          [layer5/add]                                            [layer5/softmax]                    \n"
            + "12   layer5/loss/ClipByValue           ClipByValue      [layer5/softmax]                                        [layer5/loss/ClipByValue]           \n"
            + "13   layer5/loss/log                   Log              [layer5/loss/ClipByValue]                               [layer5/loss/log]                   \n"
            + "14   layer5/loss/multiply              MulOp            [layer5/loss/log, labels]                               [layer5/loss/multiply]              \n"
            + "15   layer5/loss/neg                   Negative         [layer5/loss/multiply]                                  [layer5/loss/neg]                   \n"
            + "16   layer5/loss/reduce_sum            Sum              [layer5/loss/neg]                                       [layer5/loss/reduce_sum]            \n"
            + "17   layer5/loss/size_at               SizeAt           [labels]                                                [layer5/loss/size_at]               \n"
            + "18   layer5/loss/cast                  Cast             [layer5/loss/size_at]                                   [layer5/loss/cast]                  \n"
            + "19   layer5/loss/divide                DivOp            [layer5/loss/reduce_sum, layer5/loss/cast]              [loss]                              \n";

    @Override
    public DataType getDataType() {
        return DataType.FLOAT;
    }

    public static void testSameDiffInference(MultiLayerNetwork network, SameDiff sameDiff, INDArray input, String name){
        INDArray dl4j = network.output(input);
        INDArray sd = sameDiff.batchOutput()
                .input("input", input)
                .output(sameDiff.outputs().get(0))
                .outputSingle();

        assertTrue("Output of DL4J and SameDiff differ for " + name, dl4j.equalsWithEps(sd, 1e-3));
    }

    public static void testSameDiffInference(ComputationGraph network, SameDiff sameDiff, INDArray input, String name){
        INDArray dl4j = network.output(input)[0];
        INDArray sd = sameDiff.batchOutput()
                .input("in", input)
                .output(sameDiff.outputs().get(0))
                .outputSingle();

        assertTrue("Output of DL4J and SameDiff differ for " + name, dl4j.equalsWithEps(sd, 1e-3));
    }

    @Test
    public void testSimple() throws IOException {
        int seed = 123;

        boolean[] useDenses = {false}; // {true, false};
        Updater[] updaters = {Updater.NONE}; // Updater.values();
        Regularization[] regularizations = {null}; //{ new L2Regularization(0.0005), new L1Regularization(0.005), new WeightDecay(0.03, true)};
        LossFunction[] lossFunctions = {LossFunction.MSE, LossFunction.L1, LossFunction.MCXENT, LossFunction.COSINE_PROXIMITY, LossFunction.HINGE,
                LossFunction.SQUARED_HINGE, LossFunction.KL_DIVERGENCE, LossFunction.MEAN_ABSOLUTE_ERROR, LossFunction.L2, LossFunction.MEAN_ABSOLUTE_PERCENTAGE_ERROR, LossFunction.MEAN_SQUARED_LOGARITHMIC_ERROR, LossFunction.POISSON, LossFunction.WASSERSTEIN};


        List<String> failures = new ArrayList<>();

        for(Updater updater : updaters) {
            for(LossFunction lossFunction : lossFunctions) {
                for (boolean useDense : useDenses) {
                    for (Regularization regularization : regularizations) {

                        if (updater == Updater.CUSTOM)
                            continue;

                        IUpdater iUpdater = updater.getIUpdaterWithDefaultConfig();

                        log.info("Test with {}, {}, {}, and {}", useDense ? "dense layer" : "no dense layer",
                                regularization, lossFunction, iUpdater);

                        try {
                            Nd4j.getRandom().setSeed(seed);

                            ListBuilder partial = new NeuralNetConfiguration.Builder()
                                    .seed(seed)
                                    .updater(iUpdater)
                                    .regularization(regularization != null ? Collections.singletonList(regularization)
                                            : Collections.<Regularization>emptyList())
                                    .regularizationBias(
                                            regularization != null ? Collections.singletonList(regularization)
                                                    : Collections.<Regularization>emptyList())
                                    .list();

                            if (useDense)
                                partial.layer(new DenseLayer.Builder()
                                        .activation(Activation.RELU)
                                        .nOut(4).build());

                            MultiLayerConfiguration config = partial
                                    .layer(new OutputLayer.Builder(lossFunction).nIn(4).nOut(3).build())
                                    .setInputType(InputType.feedForward(4))
                                    .build();

                            MultiLayerNetwork network = new MultiLayerNetwork(config);
                            network.init();

                            Nd4j.getRandom().setSeed(seed);
                            SameDiff mnistSameDiff = network.toSameDiff(null, false, false);

                            assertEquals("More than one output", 1, mnistSameDiff.outputs().size());
                            assertEquals("More than one loss", 1, mnistSameDiff.getLossVariables().size());
                            assertNotNull(mnistSameDiff.getTrainingConfig());

                            INDArray example = Nd4j.rand(5, 4);
                            DataSet ds = new DataSet(Nd4j.rand(5, 4), Nd4j.rand(5, 3));
                            DataSetIterator iter = new SingletonDataSetIterator(ds);

                            testSameDiffInference(network, mnistSameDiff, example, "Inference");

                            // --- training tests ---

                            // train DL4J first
                            network.fit(iter, 1);
                            iter.reset();

                            // copy (w/ params and updater state)

                            mnistSameDiff = network.toSameDiff(null, true, false);
                            testSameDiffInference(network, mnistSameDiff, example, "Post DL4J Training");

                            // train 2 more epochs
                            iter.reset();
                            mnistSameDiff.fit(iter, 1);

                            iter.reset();
                            network.fit(iter, 1);

                            testSameDiffInference(network, mnistSameDiff, example, "Post 2nd Training");
                        } catch (AssertionError ae) {
                            ae.printStackTrace();
                            failures.add((useDense ? "Dense Layer " : "No Dense Layer ") + " with " + regularization
                                    + ", " + lossFunction
                                    + ", and " + iUpdater);
                        }
                    }
                }
            }
        }

        log.info(" --- Failures --- ");
        for(String f : failures){
            log.info(f);
        }

        assertTrue("There were failed tests", failures.isEmpty());

    }

    @Test
    public void testConversionAndTraining() throws IOException {
        int seed = 123;
        int outputNum = 10;

        Nd4j.getRandom().setSeed(seed);

        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .l2(0.0005)
                .l2Bias(0.0005)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(1e-3))
                .list()
                .layer(new ConvolutionLayer.Builder(5, 5)
                        .stride(1,1)
                        .nOut(20)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(new SubsamplingLayer.Builder(PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(new ConvolutionLayer.Builder(5, 5)
                        .stride(1,1)
                        .nOut(50)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(new SubsamplingLayer.Builder(PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(new DenseLayer.Builder().activation(Activation.RELU)
                        .nOut(500).build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutionalFlat(28,28,1))
                .build();

        MultiLayerNetwork network = new MultiLayerNetwork(config);
        network.init();

        Nd4j.getRandom().setSeed(seed);
        SameDiff mnistSameDiff = network.toSameDiff(null, false, false);

        assertEquals("More than one output", 1, mnistSameDiff.outputs().size());
        assertEquals("More than one loss", 1, mnistSameDiff.getLossVariables().size());
        assertNotNull(mnistSameDiff.getTrainingConfig());

        assertEquals("Summaries aren't equal", expectedSummary, mnistSameDiff.summary());

        MnistDataSetIterator trainData = new MnistDataSetIterator(5, 5);

        INDArray example = trainData.next().getFeatures().dup();

        testSameDiffInference(network, mnistSameDiff, example, "Inference");


        // --- training tests ---

        // train DL4J first
        network.fit(trainData, 1);
        trainData.reset();

        // copy (w/ params and updater state)

        mnistSameDiff = network.toSameDiff(null, false, false);
        testSameDiffInference(network, mnistSameDiff, example, "Post DL4J Training");


        // train 2 more epochs
        trainData.reset();
        mnistSameDiff.fit(trainData, 1);

        trainData.reset();
        network.fit(trainData, 1);

        testSameDiffInference(network, mnistSameDiff, example, "Post 2nd Training");
    }

    @Test
    public void testConversionAndTrainingGraph() throws IOException {
        int seed = 123;
        int outputNum = 10;

        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(seed)
//                .l2(0.0005)
//                .l2Bias(0.0005)
//                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(1e-3))
                .list()
                .layer(new ConvolutionLayer.Builder(5, 5)
                        .stride(1,1)
                        .nOut(20)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(new SubsamplingLayer.Builder(PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(new ConvolutionLayer.Builder(5, 5)
                        .stride(1,1)
                        .nOut(50)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(new SubsamplingLayer.Builder(PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(new DenseLayer.Builder().activation(Activation.RELU)
                        .nOut(500).build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutionalFlat(28,28,1))
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(config);
        net.init();

        ComputationGraph graph = net.toComputationGraph();
        graph.init();

        Map<String, InputType> inputTypes = new HashMap<>();
        inputTypes.put("in", InputType.convolutionalFlat(28,28,1));
        SameDiff mnistSameDiff = graph.toSameDiff(inputTypes, true, true);

        assertEquals("More than one output", 1, mnistSameDiff.outputs().size());
        assertEquals("More than one loss", 1, mnistSameDiff.getLossVariables().size());
        assertNotNull(mnistSameDiff.getTrainingConfig());

        MnistDataSetIterator trainData = new MnistDataSetIterator(10, 10);

        INDArray example = trainData.next().getFeatures().dup();

        testSameDiffInference(graph, mnistSameDiff, example, "Inference");


        // --- training tests ---

        // train DL4J first
        graph.fit(trainData, 2);
        trainData.reset();

        // copy (w/ params and updater state)

        mnistSameDiff = graph.toSameDiff(inputTypes, true, false);
        testSameDiffInference(graph, mnistSameDiff, example, "Post DL4J Training");


        // train 2 more epochs
        trainData.reset();
        mnistSameDiff.fit(trainData, 2);

        trainData.reset();
        graph.fit(trainData, 2);

        testSameDiffInference(graph, mnistSameDiff, example, "Post 2nd Training");
    }
}
