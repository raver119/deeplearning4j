package org.nd4j.autodiff.samediff.optimize;

import org.nd4j.autodiff.samediff.ArrayHolder;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.autodiff.samediff.array.OptimizedGraphArrayHolder;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.function.Supplier;

public class OptimizationHelper {

    private final SameDiff originalGraph;
    private boolean setConstantHolder = false;
    private boolean setVariableHolder = false;

    public OptimizationHelper(SameDiff originalGraph){
        this.originalGraph = originalGraph;
    }

    public OptimizationHelper arrayRecoveryFunction(String arrayName, Supplier<INDArray> fn){
        SDVariable v = originalGraph.getVariable(arrayName);
        Preconditions.checkState(v.getVariableType() == VariableType.VARIABLE || v.getVariableType() == VariableType.CONSTANT,
                "Can only set an array recovery function for a variable or a constant");

        if(v.getVariableType() == VariableType.VARIABLE){
            ArrayHolder h = originalGraph.getVariablesArrays();
            if(!setVariableHolder){
                originalGraph.setVariablesArrays(new OptimizedGraphArrayHolder(h));
                h = originalGraph.getVariablesArrays();
                setVariableHolder = true;
            }
            ((OptimizedGraphArrayHolder)h).setFunction(arrayName, fn);
        } else {
            ArrayHolder h = originalGraph.getConstantArrays();
            if(!setConstantHolder){
                originalGraph.setConstantArrays(new OptimizedGraphArrayHolder(h));
                h = originalGraph.getConstantArrays();
                setConstantHolder = true;
            }
            ((OptimizedGraphArrayHolder)h).setFunction(arrayName, fn);
        }

        return this;
    }

}
