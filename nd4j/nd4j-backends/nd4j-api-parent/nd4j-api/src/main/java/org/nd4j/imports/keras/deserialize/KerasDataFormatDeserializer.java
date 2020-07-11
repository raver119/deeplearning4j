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

package org.nd4j.imports.keras.deserialize;

import java.io.IOException;
import org.nd4j.enums.DataFormat;
import org.nd4j.imports.keras.KerasDataFormat;
import org.nd4j.shade.jackson.core.JsonParser;
import org.nd4j.shade.jackson.core.JsonProcessingException;
import org.nd4j.shade.jackson.databind.DeserializationContext;
import org.nd4j.shade.jackson.databind.deser.std.StdDeserializer;

public class KerasDataFormatDeserializer extends StdDeserializer<KerasDataFormat> {

    public KerasDataFormatDeserializer(){
        super(DataFormat.class);
    }

    @Override
    public KerasDataFormat deserialize(JsonParser jsonParser, DeserializationContext deserializationContext)
            throws IOException, JsonProcessingException {
        String format = jsonParser.getValueAsString().toLowerCase();
        if(format.equals("channels_last"))
            return KerasDataFormat.ChannelsLast;
        else if(format.equals("channels_first"))
            return KerasDataFormat.ChannelsFirst;
        else
            throw new IllegalStateException("Unknown data type " + format + ", should be either \"channels_first\" or \"channels_last\".");
    }
}
