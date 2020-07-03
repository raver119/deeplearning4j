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

import com.google.common.collect.MapMaker;
import com.google.common.collect.Maps;
import java.io.IOException;
import java.util.Arrays;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.math3.ml.neuralnet.MapUtils;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.LossLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.updater.BaseMultiLayerUpdater;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.util.ToSameDiffUtils;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import org.nd4j.autodiff.loss.LossReduce;
import org.nd4j.autodiff.samediff.SDIndex;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.impl.ActivationSigmoid;
import org.nd4j.linalg.activations.impl.ActivationSoftmax;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.NoOp;
import org.nd4j.linalg.lossfunctions.LossFunctions;
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
            + "- Name -                                    - Array Shape -     - Variable Type -   - Data Type-        - Output Of Function -                                - Inputs To Functions -\n"
            + "input                                       [-1, 1, 28, 28]     PLACEHOLDER         FLOAT               <none>                                                [ConvolutionLayer/inputPreprocessor/reshape]\n"
            + "ConvolutionLayer/inputPreprocessor/reshape  -                   ARRAY               FLOAT               ConvolutionLayer/inputPreprocessor/reshape(reshape)   [ConvolutionLayer/conv2d]\n"
            + "ConvolutionLayer/b                          [1, 20]             VARIABLE            FLOAT               <none>                                                [ConvolutionLayer/conv2d]\n"
            + "ConvolutionLayer/W                          [20, 1, 5, 5]       VARIABLE            FLOAT               <none>                                                [ConvolutionLayer/conv2d]\n"
            + "ConvolutionLayer/conv2d                     -                   ARRAY               FLOAT               ConvolutionLayer/conv2d(conv2d)                       [SubsamplingLayer/maxpool2d]\n"
            + "SubsamplingLayer/maxpool2d                  -                   ARRAY               FLOAT               SubsamplingLayer/maxpool2d(maxpool2d)                 [ConvolutionLayer_1/conv2d]\n"
            + "ConvolutionLayer_1/b                        [1, 50]             VARIABLE            FLOAT               <none>                                                [ConvolutionLayer_1/conv2d]\n"
            + "ConvolutionLayer_1/W                        [50, 20, 5, 5]      VARIABLE            FLOAT               <none>                                                [ConvolutionLayer_1/conv2d]\n"
            + "ConvolutionLayer_1/conv2d                   -                   ARRAY               FLOAT               ConvolutionLayer_1/conv2d(conv2d)                     [SubsamplingLayer_1/maxpool2d]\n"
            + "SubsamplingLayer_1/maxpool2d                -                   ARRAY               FLOAT               SubsamplingLayer_1/maxpool2d(maxpool2d)               [DenseLayer/inputPreprocessor/reshape]\n"
            + "DenseLayer/inputPreprocessor/reshape        -                   ARRAY               FLOAT               DenseLayer/inputPreprocessor/reshape(reshape)         [DenseLayer/mmul]   \n"
            + "DenseLayer/W                                [800, 500]          VARIABLE            FLOAT               <none>                                                [DenseLayer/mmul]   \n"
            + "DenseLayer/b                                [1, 500]            VARIABLE            FLOAT               <none>                                                [DenseLayer/add]    \n"
            + "DenseLayer/mmul                             -                   ARRAY               FLOAT               DenseLayer/mmul(mmul)                                 [DenseLayer/add]    \n"
            + "DenseLayer/add                              -                   ARRAY               FLOAT               DenseLayer/add(add)                                   [DenseLayer/relu]   \n"
            + "DenseLayer/relu                             -                   ARRAY               FLOAT               DenseLayer/relu(relu)                                 [OutputLayer/mmul]  \n"
            + "OutputLayer/W                               [500, 10]           VARIABLE            FLOAT               <none>                                                [OutputLayer/mmul]  \n"
            + "OutputLayer/b                               [1, 10]             VARIABLE            FLOAT               <none>                                                [OutputLayer/add]   \n"
            + "OutputLayer/mmul                            -                   ARRAY               FLOAT               OutputLayer/mmul(mmul)                                [OutputLayer/add]   \n"
            + "OutputLayer/add                             -                   ARRAY               FLOAT               OutputLayer/add(add)                                  [OutputLayer/softmax]\n"
            + "OutputLayer/softmax                         -                   ARRAY               FLOAT               OutputLayer/softmax(softmax)                          [LossNegativeLogLikelihood/ClipByValue]\n"
            + "labels                                      [-1, 10]            PLACEHOLDER         FLOAT               <none>                                                [LossNegativeLogLikelihood/multiply]\n"
            + "LossNegativeLogLikelihood/ClipByValue       -                   ARRAY               FLOAT               LossNegativeLogLikelihood/ClipByValue(ClipByValue)    [LossNegativeLogLikelihood/log]\n"
            + "LossNegativeLogLikelihood/log               -                   ARRAY               FLOAT               LossNegativeLogLikelihood/log(log)                    [LossNegativeLogLikelihood/multiply]\n"
            + "LossNegativeLogLikelihood/multiply          -                   ARRAY               FLOAT               LossNegativeLogLikelihood/multiply(multiply)          [LossNegativeLogLikelihood/neg]\n"
            + "LossNegativeLogLikelihood/neg               -                   ARRAY               FLOAT               LossNegativeLogLikelihood/neg(neg)                    [LossNegativeLogLikelihood/reduce_sum, LossNegativeLogLikelihood/shape_of]\n"
            + "LossNegativeLogLikelihood/reduce_sum        -                   ARRAY               FLOAT               LossNegativeLogLikelihood/reduce_sum(reduce_sum)      [LossNegativeLogLikelihood/divide]\n"
            + "LossNegativeLogLikelihood/shape_of          -                   ARRAY               LONG                LossNegativeLogLikelihood/shape_of(shape_of)          [LossNegativeLogLikelihood/stridedslice]\n"
            + "LossNegativeLogLikelihood/stridedslice      -                   ARRAY               LONG                LossNegativeLogLikelihood/stridedslice(stridedslice)  [LossNegativeLogLikelihood/divide]\n"
            + "loss                                        -                   ARRAY               FLOAT               LossNegativeLogLikelihood/divide(divide)                                  \n"
            + "\n"
            + "\n"
            + "--- Functions ---\n"
            + "     - Function Name -                           - Op -           - Inputs -                                                                            - Outputs -                                   \n"
            + "0    ConvolutionLayer/inputPreprocessor/reshape  Reshape          [input]                                                                               [ConvolutionLayer/inputPreprocessor/reshape]  \n"
            + "1    ConvolutionLayer/conv2d                     Conv2D           [ConvolutionLayer/inputPreprocessor/reshape, ConvolutionLayer/W, ConvolutionLayer/b]  [ConvolutionLayer/conv2d]                     \n"
            + "2    SubsamplingLayer/maxpool2d                  MaxPooling2D     [ConvolutionLayer/conv2d]                                                             [SubsamplingLayer/maxpool2d]                  \n"
            + "3    ConvolutionLayer_1/conv2d                   Conv2D           [SubsamplingLayer/maxpool2d, ConvolutionLayer_1/W, ConvolutionLayer_1/b]              [ConvolutionLayer_1/conv2d]                   \n"
            + "4    SubsamplingLayer_1/maxpool2d                MaxPooling2D     [ConvolutionLayer_1/conv2d]                                                           [SubsamplingLayer_1/maxpool2d]                \n"
            + "5    DenseLayer/inputPreprocessor/reshape        Reshape          [SubsamplingLayer_1/maxpool2d]                                                        [DenseLayer/inputPreprocessor/reshape]        \n"
            + "6    DenseLayer/mmul                             Mmul             [DenseLayer/inputPreprocessor/reshape, DenseLayer/W]                                  [DenseLayer/mmul]                             \n"
            + "7    DenseLayer/add                              AddOp            [DenseLayer/mmul, DenseLayer/b]                                                       [DenseLayer/add]                              \n"
            + "8    DenseLayer/relu                             RectifiedLinear  [DenseLayer/add]                                                                      [DenseLayer/relu]                             \n"
            + "9    OutputLayer/mmul                            Mmul             [DenseLayer/relu, OutputLayer/W]                                                      [OutputLayer/mmul]                            \n"
            + "10   OutputLayer/add                             AddOp            [OutputLayer/mmul, OutputLayer/b]                                                     [OutputLayer/add]                             \n"
            + "11   OutputLayer/softmax                         SoftMax          [OutputLayer/add]                                                                     [OutputLayer/softmax]                         \n"
            + "12   LossNegativeLogLikelihood/ClipByValue       ClipByValue      [OutputLayer/softmax]                                                                 [LossNegativeLogLikelihood/ClipByValue]       \n"
            + "13   LossNegativeLogLikelihood/log               Log              [LossNegativeLogLikelihood/ClipByValue]                                               [LossNegativeLogLikelihood/log]               \n"
            + "14   LossNegativeLogLikelihood/multiply          MulOp            [LossNegativeLogLikelihood/log, labels]                                               [LossNegativeLogLikelihood/multiply]          \n"
            + "15   LossNegativeLogLikelihood/neg               Negative         [LossNegativeLogLikelihood/multiply]                                                  [LossNegativeLogLikelihood/neg]               \n"
            + "16   LossNegativeLogLikelihood/reduce_sum        Sum              [LossNegativeLogLikelihood/neg]                                                       [LossNegativeLogLikelihood/reduce_sum]        \n"
            + "17   LossNegativeLogLikelihood/shape_of          Shape            [LossNegativeLogLikelihood/neg]                                                       [LossNegativeLogLikelihood/shape_of]          \n"
            + "18   LossNegativeLogLikelihood/stridedslice      StridedSlice     [LossNegativeLogLikelihood/shape_of]                                                  [LossNegativeLogLikelihood/stridedslice]      \n"
            + "19   LossNegativeLogLikelihood/divide            DivOp            [LossNegativeLogLikelihood/reduce_sum, LossNegativeLogLikelihood/stridedslice]        [loss]                                        \n";

    @Override
    public DataType getDataType() {
        return DataType.FLOAT;
    }

    public static void testSameDiffInference(MultiLayerNetwork network, INDArray input){
        SameDiff sameDiff = network.toSameDiff(null, true, true);
        INDArray dl4j = network.output(input);
        INDArray sd = sameDiff.batchOutput()
                .input("input", input)
                .output(sameDiff.outputs().get(0))
                .outputSingle();

        assertTrue(dl4j.equalsWithEps(sd, 1e-3));
    }

    @Test
    public void testConversion() throws IOException {
        int seed = 123;
        int outputNum = 10;

        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .l2(0.0005)
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

        SameDiff mnistSameDiff = network.toSameDiff(null, true, true);

        assertEquals("More than one output", 1, mnistSameDiff.outputs().size());
        assertEquals("More than one loss", 1, mnistSameDiff.getLossVariables().size());
        assertNotNull(mnistSameDiff.getTrainingConfig());

        assertEquals("Summaries aren't equal", expectedSummary, mnistSameDiff.summary());

        MnistDataSetIterator trainData = new MnistDataSetIterator(10, 100);

        INDArray example = trainData.next().getFeatures();

        testSameDiffInference(network, example);

        //TODO test output dims of mseLoss

        // training
        //TODO needs a crossentropy op
//        trainData.reset();
//
//        mnistSameDiff.fit(trainData, 2);
//
//        network.fit(trainData, 2);
//
//        trainData.reset();
//        example = trainData.next().getFeatures();
//
//        // post training test
//
//        testSameDiffInference(network, example);
    }
}
