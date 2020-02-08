package org.nd4j.autodiff.optimization.util;

import lombok.Builder;
import lombok.Data;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.optimize.Optimizer;
import org.nd4j.autodiff.samediff.optimize.OptimizerSet;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.File;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Data
public class OptTestConfig {

    private SameDiff original;
    private Map<String, INDArray> placeholders;
    private List<String> outputs;
    private File tempFolder;
    private Map<String,Class<? extends Optimizer>> mustApply;
    private List<OptimizerSet> optimizerSets;

    public static Builder builder(){
        return new Builder();
    }

    public static class Builder {

        private SameDiff original;
        private Map<String, INDArray> placeholders;
        private List<String> outputs;
        private File tempFolder;
        private Map<String,Class<? extends Optimizer>> mustApply;
        private List<OptimizerSet> optimizerSets;

        public Builder original(SameDiff sd){
            original = sd;
            return this;
        }

        public Builder placeholder(String ph, INDArray arr){
            if(placeholders == null)
                placeholders = new HashMap<>();
            placeholders.put(ph, arr);
            return this;
        }

        public Builder placeholders(Map<String,INDArray> map){
            placeholders = map;
            return this;
        }

        public Builder outputs(String... outputs){
            this.outputs = Arrays.asList(outputs);
            return this;
        }

        public Builder outputs(List<String> outputs){
            this.outputs = outputs;
            return this;
        }

        public Builder mustApply(String opName, Class<? extends Optimizer> optimizerClass){
            if(mustApply == null)
                mustApply = new HashMap<>();
            mustApply.put(opName, optimizerClass);
            return this;
        }

        public Builder optimizerSets(List<OptimizerSet> list){
            this.optimizerSets = list;
            return this;
        }

        public OptTestConfig build(){
            OptTestConfig c = new OptTestConfig();
            c.original = original;
            c.placeholders = placeholders;
            c.outputs = outputs;
            c.tempFolder = tempFolder;
            c.mustApply = mustApply;
            c.optimizerSets = optimizerSets;
            return c;
        }

    }

}
