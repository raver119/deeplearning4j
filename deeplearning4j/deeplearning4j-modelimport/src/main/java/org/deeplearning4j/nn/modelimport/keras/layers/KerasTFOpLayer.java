package org.deeplearning4j.nn.modelimport.keras.layers;

import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.TFOpLayer;
import org.deeplearning4j.nn.modelimport.keras.KerasLayer;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;

import java.util.Map;


public class KerasTFOpLayer extends KerasLayer {

    public KerasTFOpLayer(Integer kerasVersion) throws UnsupportedKerasConfigurationException {
        super(kerasVersion);
        if (kerasVersion != 2){
            throw new UnsupportedKerasConfigurationException("KerasTFOpLayer expects Keras version 2");
        }
    }

    /**
     * Constructor from parsed Keras layer configuration dictionary.
     *
     * @param layerConfig dictionary containing Keras layer configuration
     * @throws InvalidKerasConfigurationException     Invalid Keras config
     * @throws UnsupportedKerasConfigurationException Unsupported Keras config
     */
    public KerasTFOpLayer(Map<String, Object> layerConfig)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        this(layerConfig, true);
    }

    /**
     * Constructor from parsed Keras layer configuration dictionary.
     *
     * @param layerConfig           dictionary containing Keras layer configuration
     * @param enforceTrainingConfig whether to enforce training-related configuration options
     * @throws InvalidKerasConfigurationException     Invalid Keras config
     * @throws UnsupportedKerasConfigurationException Unsupported Keras config
     */
    public KerasTFOpLayer(Map<String, Object> layerConfig, boolean enforceTrainingConfig) throws UnsupportedKerasConfigurationException, InvalidKerasConfigurationException{
        super(layerConfig, enforceTrainingConfig);
        for(Map.Entry<String, Object> e: layerConfig.entrySet()){
            System.out.println(e.getKey() + " : " + e.getValue());
        }
        this.layer = new TFOpLayer((Map)((Map)layerConfig.get("config")).get("node_def"), (Map)((Map)layerConfig.get("config")).get("constants"));
    }

    /**
     * Get layer output type.
     *
     * @param inputType Array of InputTypes
     * @return output type as InputType
     * @throws InvalidKerasConfigurationException Invalid Keras configuration
     */
    public InputType getOutputType(InputType... inputType){
        return this.layer.getOutputType(0, inputType[0]);
    }



}
