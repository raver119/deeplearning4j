package org.deeplearning4j.nn.modelimport.keras;

import org.deeplearning4j.BaseDL4JTest;
import org.junit.Test;

import java.io.File;

public class TFKerasTests extends BaseDL4JTest{

    @Test
    public void testModelWithTFOp() throws Exception{
        File f = new File("C:\\Users\\fariz\\desktop\\model.h5");
        KerasModelImport.importKerasModelAndWeights(f.getAbsolutePath());
    }

}
