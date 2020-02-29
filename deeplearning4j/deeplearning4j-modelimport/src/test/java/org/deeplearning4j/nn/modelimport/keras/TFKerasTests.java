package org.deeplearning4j.nn.modelimport.keras;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.junit.Assert;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;

public class TFKerasTests extends BaseDL4JTest{

    @Test
    public void testModelWithTFOp() throws Exception{
        File f = new File("C:\\Users\\fariz\\desktop\\model.h5");
       ComputationGraph graph = KerasModelImport.importKerasModelAndWeights(f.getAbsolutePath());
        INDArray out = graph.outputSingle(Nd4j.zeros(12, 3, 2));
        Assert.assertArrayEquals(new long[]{12, 3}, out.shape());
    }

}
