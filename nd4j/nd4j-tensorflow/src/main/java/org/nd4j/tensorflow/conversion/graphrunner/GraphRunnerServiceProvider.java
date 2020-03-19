package org.nd4j.tensorflow.conversion.graphrunner;

import org.nd4j.TFGraphRunnerService;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.tensorflow.conversion.TensorDataType;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class GraphRunnerServiceProvider implements TFGraphRunnerService {

    private GraphRunner graphRunner;

    @Override
    public TFGraphRunnerService init(
            List<String> inputNames,
            List<String> outputNames,
            byte[] graphBytes,
            List<String> inputDataTypes){
        if (inputNames.size() != inputDataTypes.size()){
            throw new IllegalArgumentException("inputNames.size() != inputDataTypes.size()");
        }
        Map<String, TensorDataType> convertedDataTypes = new HashMap<>();
        for (int i = 0; i < inputNames.size(); i++){
            convertedDataTypes.put(inputNames.get(i), TensorDataType.fromProtoValue(inputDataTypes.get(i)));
        }
        graphRunner = GraphRunner.builder().inputNames(inputNames)
                .outputNames(outputNames).graphBytes(graphBytes)
                .outputDataTypes(convertedDataTypes).build();
        return this;

    }

    @Override
    public Map<String, INDArray> run(Map<String, INDArray> inputs){
        if (graphRunner == null){
            throw new RuntimeException("GraphRunner not initialized.");
        }
        return graphRunner.run(inputs);
    }
}
