package org.nd4j.autodiff.samediff.optimize.optimizations;

import lombok.NonNull;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.autodiff.samediff.internal.Variable;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

public class OptimizationUtils {

    private OptimizationUtils(){ }

    public static void replaceOpInputsWith(SameDiff sd, @NonNull String replaceInput, @NonNull String newInput){
        if(replaceInput.equals(newInput))
            return;

        //Update op input structure: Replace all instances replaceInput->X with newInput->X
        Collection<SameDiffOp> ops = sd.getOps().values();
        for(SameDiffOp o : ops){
            List<String> l = o.getInputsToOp();
            while(l != null && l.contains(replaceInput)){
                int idx = l.indexOf(replaceInput);
                l.set(idx, newInput);
            }
        }

        //Update variable structure
        Variable v = sd.getVariables().get(replaceInput);
        Variable v2 = sd.getVariables().get(newInput);
        //NOTE: this only works if we carefully control the order in which replaceOpInputsWith is called!
        v2.setInputsForOp(v.getInputsForOp());
        v.setInputsForOp(new ArrayList<String>());
    }

    public static void removeOp(@NonNull SameDiff sd, @NonNull String opToRemove){
        SameDiffOp op = sd.getOps().remove(opToRemove);
        for(String s : op.getInputsToOp()){
            Variable v = sd.getVariables().get(s);
            v.getInputsForOp().remove(op.getName());
        }
    }

}
