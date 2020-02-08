package org.nd4j.autodiff.optimization.util;

import lombok.Getter;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.autodiff.samediff.optimize.Optimizer;
import org.nd4j.autodiff.samediff.optimize.debug.OptimizationDebugger;

import java.util.HashMap;
import java.util.Map;

public class OptimizationRecordingDebugger implements OptimizationDebugger {

    @Getter
    private Map<String,Optimizer> applied = new HashMap<>();

    @Override
    public void beforeOptimizationCheck(SameDiff sd, SameDiffOp op, Optimizer o) {
        //No op
    }

    @Override
    public void afterOptimizationsCheck(SameDiff sd, SameDiffOp op, Optimizer o, boolean wasApplied) {
        if(wasApplied){
            applied.put(op.getName(), o);
        }
    }
}
