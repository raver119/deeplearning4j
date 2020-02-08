package org.nd4j.autodiff.optimization;

import org.junit.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.autodiff.samediff.optimize.GraphOptimizer;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.util.Collections;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;

public class TestOptimization extends BaseNd4jTest {

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

        SameDiff sd = SameDiff.create();
        SDVariable c = sd.constant("c", Nd4j.scalar(1.0));
        SDVariable c2 = c.add("add", 1);
        SDVariable v = sd.var("variable", Nd4j.scalar(1.0));
        SDVariable out = v.sub("out", c2);

        SameDiff optimized = GraphOptimizer.optimize(sd);
        assertEquals(3, optimized.getVariables().size());       //"add", "variable", "out" -> "c" should be removed
        assertEquals(VariableType.CONSTANT, optimized.getVariable("add").getVariableType());
        assertEquals(1, optimized.getOps().size());
        assertEquals("subtract", optimized.getOps().values().iterator().next().getName());

        assertFalse(optimized.hasVariable("c"));

        assertEquals(sd.outputSingle(Collections.emptyMap(), "out"), optimized.outputSingle(Collections.emptyMap(), "out"));
    }
}
