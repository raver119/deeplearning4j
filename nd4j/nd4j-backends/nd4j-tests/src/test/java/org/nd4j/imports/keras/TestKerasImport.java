/*
 * ******************************************************************************
 *  * Copyright (c) 2020 Konduit K.K.
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.imports.keras;

import org.junit.Test;
import org.nd4j.imports.keras.layers.KerasLayer;
import org.nd4j.imports.keras.models.Sequential;
import org.nd4j.shade.jackson.core.JsonProcessingException;
import org.nd4j.shade.jackson.databind.ObjectMapper;

public class TestKerasImport {

    @Test
    public void test() throws JsonProcessingException {


        String json = "{\"class_name\": \"Sequential\", \"config\": [{\"class_name\": \"Conv2D\", \"config\": {\"name\": \"conv2d_1\", \"trainable\": true, \"batch_input_shape\": [null, 28, 28, 1], \"dtype\": \"float32\", \"filters\": 16, \"kernel_size\": [3, 3], \"strides\": [2, 2], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"linear\", \"use_bias\": true, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 1.0, \"mode\": \"fan_avg\", \"distribution\": \"uniform\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}}, {\"class_name\": \"LeakyReLU\", \"config\": {\"name\": \"leaky_re_lu_1\", \"trainable\": true, \"alpha\": 0.20000000298023224}}, {\"class_name\": \"Dropout\", \"config\": {\"name\": \"dropout_1\", \"trainable\": true, \"rate\": 0.25, \"noise_shape\": null, \"seed\": null}}, {\"class_name\": \"Conv2D\", \"config\": {\"name\": \"conv2d_2\", \"trainable\": true, \"filters\": 32, \"kernel_size\": [3, 3], \"strides\": [2, 2], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"linear\", \"use_bias\": true, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 1.0, \"mode\": \"fan_avg\", \"distribution\": \"uniform\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}}, {\"class_name\": \"ZeroPadding2D\", \"config\": {\"name\": \"zero_padding2d_1\", \"trainable\": true, \"padding\": [[0, 1], [0, 1]], \"data_format\": \"channels_last\"}}, {\"class_name\": \"LeakyReLU\", \"config\": {\"name\": \"leaky_re_lu_2\", \"trainable\": true, \"alpha\": 0.20000000298023224}}, {\"class_name\": \"Dropout\", \"config\": {\"name\": \"dropout_2\", \"trainable\": true, \"rate\": 0.25, \"noise_shape\": null, \"seed\": null}}, {\"class_name\": \"BatchNormalization\", \"config\": {\"name\": \"batch_normalization_1\", \"trainable\": true, \"axis\": -1, \"momentum\": 0.8, \"epsilon\": 0.001, \"center\": true, \"scale\": true, \"beta_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"gamma_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"moving_mean_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"moving_variance_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"beta_regularizer\": null, \"gamma_regularizer\": null, \"beta_constraint\": null, \"gamma_constraint\": null}}, {\"class_name\": \"Conv2D\", \"config\": {\"name\": \"conv2d_3\", \"trainable\": true, \"filters\": 64, \"kernel_size\": [3, 3], \"strides\": [2, 2], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"linear\", \"use_bias\": true, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 1.0, \"mode\": \"fan_avg\", \"distribution\": \"uniform\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}}, {\"class_name\": \"LeakyReLU\", \"config\": {\"name\": \"leaky_re_lu_3\", \"trainable\": true, \"alpha\": 0.20000000298023224}}, {\"class_name\": \"Dropout\", \"config\": {\"name\": \"dropout_3\", \"trainable\": true, \"rate\": 0.25, \"noise_shape\": null, \"seed\": null}}, {\"class_name\": \"BatchNormalization\", \"config\": {\"name\": \"batch_normalization_2\", \"trainable\": true, \"axis\": -1, \"momentum\": 0.8, \"epsilon\": 0.001, \"center\": true, \"scale\": true, \"beta_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"gamma_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"moving_mean_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"moving_variance_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"beta_regularizer\": null, \"gamma_regularizer\": null, \"beta_constraint\": null, \"gamma_constraint\": null}}, {\"class_name\": \"Conv2D\", \"config\": {\"name\": \"conv2d_4\", \"trainable\": true, \"filters\": 128, \"kernel_size\": [3, 3], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"linear\", \"use_bias\": true, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 1.0, \"mode\": \"fan_avg\", \"distribution\": \"uniform\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}}, {\"class_name\": \"LeakyReLU\", \"config\": {\"name\": \"leaky_re_lu_4\", \"trainable\": true, \"alpha\": 0.20000000298023224}}, {\"class_name\": \"Dropout\", \"config\": {\"name\": \"dropout_4\", \"trainable\": true, \"rate\": 0.25, \"noise_shape\": null, \"seed\": null}}, {\"class_name\": \"Flatten\", \"config\": {\"name\": \"flatten_1\", \"trainable\": true}}]}";
        Sequential test = KerasImportUtils.kerasMapper().readValue(json, Sequential.class);
        System.out.println(test);
    }
}
