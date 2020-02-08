package org.nd4j.autodiff.samediff.optimize.optimizations;

import org.nd4j.autodiff.samediff.ArrayHolder;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.autodiff.samediff.optimize.Optimizer;

import java.util.Properties;

public class IdentityFunctionOptimizations extends BaseOptimizerSet {

    /**
     * Remove permute(0,1,2,...,rank-1) as this is a no-op
     */
    public static class RemoveIdentityPermute implements Optimizer {

        @Override
        public boolean checkAndApply(SameDiff sd, Properties optimizationConfig, SameDiffOp op, ArrayHolder constantArrays, ArrayHolder variablesArrays) {
            return false;
        }
    }

    /**
     * Remove identity(x)
     */
    public static class RemoveIdentityOps implements Optimizer {
        @Override
        public boolean checkAndApply(SameDiff sd, Properties optimizationConfig, SameDiffOp op, ArrayHolder constantArrays, ArrayHolder variablesArrays) {
            return false;
        }
    }
}
