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

import java.io.IOException;
import java.util.Collection;
import java.util.HashSet;
import java.util.Set;
import org.nd4j.enums.DataFormat;
import org.nd4j.imports.keras.deserialize.KerasDataFormatDeserializer;
import org.nd4j.imports.keras.deserialize.KerasDatatypeDeserializer;
import org.nd4j.imports.keras.deserialize.KerasName;
import org.nd4j.imports.keras.deserialize.KerasPaddingDeserializer;
import org.nd4j.imports.keras.deserialize.KerasWrappedJson;
import org.nd4j.imports.keras.deserialize.KerasWrapperDeserializer;
import org.nd4j.imports.keras.layers.KerasLayer;
import org.nd4j.imports.keras.models.Sequential;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.PaddingMode;
import org.nd4j.shade.guava.reflect.ClassPath;
import org.nd4j.shade.jackson.annotation.JsonAutoDetect.Visibility;
import org.nd4j.shade.jackson.annotation.PropertyAccessor;
import org.nd4j.shade.jackson.core.JsonProcessingException;
import org.nd4j.shade.jackson.databind.BeanDescription;
import org.nd4j.shade.jackson.databind.DeserializationConfig;
import org.nd4j.shade.jackson.databind.JsonDeserializer;
import org.nd4j.shade.jackson.databind.ObjectMapper;
import org.nd4j.shade.jackson.databind.deser.BeanDeserializerModifier;
import org.nd4j.shade.jackson.databind.jsontype.NamedType;
import org.nd4j.shade.jackson.databind.module.SimpleModule;

public class KerasImportUtils {

    private static final Set<Class<? extends KerasLayer>> layers = new HashSet<>();
    static{
        try {
            Set<ClassPath.ClassInfo> classes = ClassPath.from(KerasLayer.class.getClassLoader()).getTopLevelClassesRecursive("org.nd4j.imports.keras.layers");
            for(ClassPath.ClassInfo info : classes){
                Class<?> klass = info.load();
                if(KerasLayer.class != klass && KerasLayer.class.isAssignableFrom(klass))
                    layers.add(klass.asSubclass(KerasLayer.class));
            }
        } catch (IOException e) {
            e.printStackTrace();
            throw new RuntimeException("Could not find KerasLayer subclasses", e);
        }
    }

    public static ObjectMapper kerasMapper(){
        SimpleModule module = new SimpleModule();
        module.setDeserializerModifier(new BeanDeserializerModifier() {
            @Override
            public JsonDeserializer<?> modifyDeserializer(DeserializationConfig config, BeanDescription beanDesc,
                    JsonDeserializer<?> deserializer) {
                if(beanDesc.getClassAnnotations().has(KerasWrappedJson.class) ||
                        KerasLayer.class.isAssignableFrom(beanDesc.getBeanClass()))
                    return new KerasWrapperDeserializer(deserializer, beanDesc.getBeanClass());
                return deserializer;
            }
        });

        module.addDeserializer(DataType.class, new KerasDatatypeDeserializer());
        module.addDeserializer(PaddingMode.class, new KerasPaddingDeserializer());
        module.addDeserializer(DataFormat.class, new KerasDataFormatDeserializer());

        ObjectMapper mapper = new ObjectMapper();
        mapper.registerModule(module);

        for(Class<? extends KerasLayer> klass : layers){
            String className;
            if(klass.isAnnotationPresent(KerasName.class))
                className = klass.getAnnotation(KerasName.class).name();
            else
                className = klass.getSimpleName();

            mapper.registerSubtypes(new NamedType(klass, className));
        }

        return mapper;
    }
}
