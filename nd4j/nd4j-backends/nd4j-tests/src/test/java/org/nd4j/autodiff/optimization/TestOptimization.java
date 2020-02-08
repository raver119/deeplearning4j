package org.nd4j.autodiff.optimization;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.nd4j.autodiff.optimization.util.OptTestConfig;
import org.nd4j.autodiff.optimization.util.OptimizationTestUtil;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.autodiff.samediff.optimize.GraphOptimizer;
import org.nd4j.autodiff.samediff.optimize.optimizations.ConstantFunctionOptimizations;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.util.Arrays;
import java.util.Collections;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;

public class TestOptimization extends BaseNd4jTest {

    @Rule
    public TemporaryFolder testDir = new TemporaryFolder();

    public TestOptimization(Nd4jBackend backend) {
        super(backend);
    }

    @Override
    public char ordering() {
        return 'c';
    }

    @Override
    public long getTimeoutMilliseconds() {
        return 1_000_000_000L;
    }

    @Test
    public void testConstantOpFolding(){
        //We expect 2 things in this test:
        //(a) the output of  add(constant, constant) is pre-calculated and itself becomes a constant
        //(b) the


        SameDiff sd = SameDiff.create();
        SDVariable c = sd.constant("c", Nd4j.scalar(1.0));
        SDVariable c2 = c.add("add", 1);
        SDVariable v = sd.var("variable", Nd4j.scalar(1.0));
        SDVariable out = v.sub("out", c2);

        SameDiff copy = sd.dup();

        SameDiff optimized = GraphOptimizer.optimize(sd);
        assertEquals(3, optimized.getVariables().size());       //"add", "variable", "out" -> "c" should be removed
        assertEquals(VariableType.CONSTANT, optimized.getVariable("add").getVariableType());
        assertEquals(1, optimized.getOps().size());
        assertEquals("subtract", optimized.getOps().values().iterator().next().getName());

        assertFalse(optimized.hasVariable("c"));

        assertEquals(sd.outputSingle(Collections.emptyMap(), "out"), optimized.outputSingle(Collections.emptyMap(), "out"));

        //Check the

        //Check that the original can be saved and loaded, and still gives the same results

    }

    @Test
    public void testConstantOpFolding2(){
        //We expect 2 things in this test:
        //(a) the output of  add(constant, constant) is pre-calculated and itself becomes a constant
        //(b) the


        SameDiff sd = SameDiff.create();
        SDVariable c = sd.constant("c", Nd4j.scalar(1.0));
        SDVariable c2 = c.add("add", 1);
        SDVariable v = sd.var("variable", Nd4j.scalar(1.0));
        SDVariable out = v.sub("out", c2);

        OptTestConfig conf = OptTestConfig.builder()
                .original(sd)
                .outputs(Collections.singletonList("out"))
                .mustApply(sd.getVariables().get("add").getOutputOfOp(), ConstantFunctionOptimizations.FoldConstantFunctions.class)
                .build();

        SameDiff optimized = OptimizationTestUtil.testOptimization(conf);
        assertEquals(3, optimized.getVariables().size());       //"add", "variable", "out" -> "c" should be removed
        assertEquals(VariableType.CONSTANT, optimized.getVariable("add").getVariableType());
        assertEquals(1, optimized.getOps().size());
        assertEquals("subtract", optimized.getOps().values().iterator().next().getName());

        assertFalse(optimized.hasVariable("c"));

        assertEquals(sd.outputSingle(Collections.emptyMap(), "out"), optimized.outputSingle(Collections.emptyMap(), "out"));

    }
}
