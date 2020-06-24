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

package org.deeplearning4j.nn.multilayer;

import static org.junit.Assert.*;

import com.google.common.collect.MapMaker;
import com.google.common.collect.Maps;
import java.io.IOException;
import java.util.Arrays;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.math3.ml.neuralnet.MapUtils;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import org.nd4j.autodiff.loss.LossReduce;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.lossfunctions.impl.LossBinaryXENT;
import org.nd4j.linalg.lossfunctions.impl.LossCosineProximity;

@Slf4j
public class ToSameDiffTest extends BaseDL4JTest {

    private static OpExecutioner.ProfilingMode origMode;

    private static final String expectedSummary = "--- Summary ---\n"
            + "Variables:               24                   (9 with arrays)\n"
            + "Functions:               13                  \n"
            + "SameDiff Function Defs:  0                   \n"
            + "Loss function variables: [weighted_cross_entropy_with_logits]\n"
            + "\n"
            + "--- Variables ---\n"
            + "- Name -                                    - Array Shape -     - Variable Type -   - Data Type-        - Output Of Function -                                                  - Inputs To Functions -\n"
            + "input                                       [-1, 1, 28, 28]     PLACEHOLDER         FLOAT               <none>                                                                  [ConvolutionLayer/inputPreprocessor/reshape]\n"
            + "ConvolutionLayer/inputPreprocessor/reshape  -                   ARRAY               FLOAT               ConvolutionLayer/inputPreprocessor/reshape(reshape)                     [ConvolutionLayer/conv2d]\n"
            + "ConvolutionLayer/b                          [1, 20]             VARIABLE            FLOAT               <none>                                                                  [ConvolutionLayer/conv2d]\n"
            + "ConvolutionLayer/W                          [20, 1, 5, 5]       VARIABLE            FLOAT               <none>                                                                  [ConvolutionLayer/conv2d]\n"
            + "ConvolutionLayer/conv2d                     -                   ARRAY               FLOAT               ConvolutionLayer/conv2d(conv2d)                                         [SubsamplingLayer/maxpool2d]\n"
            + "SubsamplingLayer/maxpool2d                  -                   ARRAY               FLOAT               SubsamplingLayer/maxpool2d(maxpool2d)                                   [ConvolutionLayer_1/conv2d]\n"
            + "ConvolutionLayer_1/b                        [1, 50]             VARIABLE            FLOAT               <none>                                                                  [ConvolutionLayer_1/conv2d]\n"
            + "ConvolutionLayer_1/W                        [50, 20, 5, 5]      VARIABLE            FLOAT               <none>                                                                  [ConvolutionLayer_1/conv2d]\n"
            + "ConvolutionLayer_1/conv2d                   -                   ARRAY               FLOAT               ConvolutionLayer_1/conv2d(conv2d)                                       [SubsamplingLayer_1/maxpool2d]\n"
            + "SubsamplingLayer_1/maxpool2d                -                   ARRAY               FLOAT               SubsamplingLayer_1/maxpool2d(maxpool2d)                                 [DenseLayer/inputPreprocessor/reshape]\n"
            + "DenseLayer/inputPreprocessor/reshape        -                   ARRAY               FLOAT               DenseLayer/inputPreprocessor/reshape(reshape)                           [DenseLayer/mmul]   \n"
            + "DenseLayer/W                                [800, 500]          VARIABLE            FLOAT               <none>                                                                  [DenseLayer/mmul]   \n"
            + "DenseLayer/b                                [1, 500]            VARIABLE            FLOAT               <none>                                                                  [DenseLayer/add]    \n"
            + "DenseLayer/mmul                             -                   ARRAY               FLOAT               DenseLayer/mmul(mmul)                                                   [DenseLayer/add]    \n"
            + "DenseLayer/add                              -                   ARRAY               FLOAT               DenseLayer/add(add)                                                     [DenseLayer/relu]   \n"
            + "DenseLayer/relu                             -                   ARRAY               FLOAT               DenseLayer/relu(relu)                                                   [OutputLayer/mmul]  \n"
            + "OutputLayer/W                               [500, 10]           VARIABLE            FLOAT               <none>                                                                  [OutputLayer/mmul]  \n"
            + "OutputLayer/b                               [1, 10]             VARIABLE            FLOAT               <none>                                                                  [OutputLayer/add]   \n"
            + "OutputLayer/mmul                            -                   ARRAY               FLOAT               OutputLayer/mmul(mmul)                                                  [OutputLayer/add]   \n"
            + "OutputLayer/add                             -                   ARRAY               FLOAT               OutputLayer/add(add)                                                    [OutputLayer/softmax]\n"
            + "OutputLayer/softmax                         -                   ARRAY               FLOAT               OutputLayer/softmax(softmax)                                            [weighted_cross_entropy_with_logits]\n"
            + "labels                                      [-1, 10]            PLACEHOLDER         FLOAT               <none>                                                                  [weighted_cross_entropy_with_logits]\n"
            + "sd_var                                      []                  CONSTANT            INT                 <none>                                                                  [weighted_cross_entropy_with_logits]\n"
            + "weighted_cross_entropy_with_logits          -                   ARRAY               FLOAT               weighted_cross_entropy_with_logits(weighted_cross_entropy_with_logits)                      \n"
            + "\n"
            + "\n"
            + "--- Functions ---\n"
            + "     - Function Name -                           - Op -                    - Inputs -                                                                            - Outputs -                                   \n"
            + "0    ConvolutionLayer/inputPreprocessor/reshape  Reshape                   [input]                                                                               [ConvolutionLayer/inputPreprocessor/reshape]  \n"
            + "1    ConvolutionLayer/conv2d                     Conv2D                    [ConvolutionLayer/inputPreprocessor/reshape, ConvolutionLayer/W, ConvolutionLayer/b]  [ConvolutionLayer/conv2d]                     \n"
            + "2    SubsamplingLayer/maxpool2d                  MaxPooling2D              [ConvolutionLayer/conv2d]                                                             [SubsamplingLayer/maxpool2d]                  \n"
            + "3    ConvolutionLayer_1/conv2d                   Conv2D                    [SubsamplingLayer/maxpool2d, ConvolutionLayer_1/W, ConvolutionLayer_1/b]              [ConvolutionLayer_1/conv2d]                   \n"
            + "4    SubsamplingLayer_1/maxpool2d                MaxPooling2D              [ConvolutionLayer_1/conv2d]                                                           [SubsamplingLayer_1/maxpool2d]                \n"
            + "5    DenseLayer/inputPreprocessor/reshape        Reshape                   [SubsamplingLayer_1/maxpool2d]                                                        [DenseLayer/inputPreprocessor/reshape]        \n"
            + "6    DenseLayer/mmul                             Mmul                      [DenseLayer/inputPreprocessor/reshape, DenseLayer/W]                                  [DenseLayer/mmul]                             \n"
            + "7    DenseLayer/add                              AddOp                     [DenseLayer/mmul, DenseLayer/b]                                                       [DenseLayer/add]                              \n"
            + "8    DenseLayer/relu                             RectifiedLinear           [DenseLayer/add]                                                                      [DenseLayer/relu]                             \n"
            + "9    OutputLayer/mmul                            Mmul                      [DenseLayer/relu, OutputLayer/W]                                                      [OutputLayer/mmul]                            \n"
            + "10   OutputLayer/add                             AddOp                     [OutputLayer/mmul, OutputLayer/b]                                                     [OutputLayer/add]                             \n"
            + "11   OutputLayer/softmax                         SoftMax                   [OutputLayer/add]                                                                     [OutputLayer/softmax]                         \n"
            + "12   weighted_cross_entropy_with_logits          WeightedCrossEntropyLoss  [labels, OutputLayer/softmax, sd_var]                                                 [weighted_cross_entropy_with_logits]          \n";

    @BeforeClass
    public static void beforeClass(){
        origMode = Nd4j.getExecutioner().getProfilingMode();
    }

    @Before
    public void before() {
        Nd4j.getExecutioner().setProfilingMode(OpExecutioner.ProfilingMode.SCOPE_PANIC);
    }

    @AfterClass
    public static void afterClass() {
        Nd4j.getExecutioner().setProfilingMode(origMode);
    }

    @Override
    public DataType getDataType() {
        return DataType.FLOAT;
    }

    public static void testSameDiffInference(MultiLayerNetwork network, INDArray input){
        SameDiff sameDiff = network.toSameDiff();
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

        SameDiff mnistSameDiff = network.toSameDiff();

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

    @Test
    public void testMSE(){
        SameDiff sd = SameDiff.create();

        System.out.println(sd.nn.relu(sd.constant(1), 2).eval());

    }
}
