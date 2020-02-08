package org.nd4j.autodiff.samediff.optimize.debug;

import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.autodiff.samediff.optimize.Optimizer;

public interface OptimizationDebugger {

    void beforeOptimizationCheck(SameDiff sd, SameDiffOp op, Optimizer o);

    void afterOptimizationsCheck(SameDiff sd, SameDiffOp op, Optimizer o, boolean wasApplied);

}
