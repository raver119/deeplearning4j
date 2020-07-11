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

import static org.nd4j.imports.keras.deserialize.KerasWrapperDeserializer.CLASS_PROP_NAME;
import static org.nd4j.imports.keras.deserialize.KerasWrapperDeserializer.CONFIG_PROP_NAME;

import java.io.IOException;
import org.nd4j.imports.keras.KerasNode;
import org.nd4j.imports.keras.KerasNodeIndex;
import org.nd4j.imports.keras.layers.KerasLayer;
import org.nd4j.shade.jackson.core.JsonParser;
import org.nd4j.shade.jackson.core.JsonProcessingException;
import org.nd4j.shade.jackson.databind.DeserializationContext;
import org.nd4j.shade.jackson.databind.JsonNode;
import org.nd4j.shade.jackson.databind.deser.std.StdDeserializer;
import org.nd4j.shade.jackson.databind.node.ObjectNode;

public class KerasNodeDeserializer extends StdDeserializer<KerasNode> {

    public KerasNodeDeserializer(){
        super(KerasNode.class);
    }

    @Override
    public KerasNode deserialize(JsonParser jsonParser, DeserializationContext deserializationContext)
            throws IOException, JsonProcessingException {

        // structure is the same as a Keras wrapper, but with inbound_nodes.  Need to pull inbound_nodes out and pass the rest through to Layer.
        JsonNode tree = (JsonNode) jsonParser.readValueAsTree();

        KerasNodeIndex[][] invocations = jsonParser.getCodec().treeToValue(tree.get("inbound_nodes"), KerasNodeIndex[][].class);
        String name = jsonParser.getCodec().treeToValue(tree.get("name"), String.class);

        ObjectNode layerNode = deserializationContext.getNodeFactory().objectNode();
        layerNode.set(CONFIG_PROP_NAME, tree.get(CONFIG_PROP_NAME));
        layerNode.set(CLASS_PROP_NAME, tree.get(CLASS_PROP_NAME));

        KerasLayer layer = jsonParser.getCodec().treeToValue(layerNode, KerasLayer.class);

        return new KerasNode(name, layer, invocations);
    }
}
