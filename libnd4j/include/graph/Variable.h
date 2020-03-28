/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
// @author raver119@gmail.com
//

#ifndef LIBND4J_VARIABLE_H
#define LIBND4J_VARIABLE_H

#include <string>
#include <array/NDArray.h>
#include <array/NDArrayList.h>
#include <graph/VariableType.h>
#include <graph/generated/array_generated.h>
#include <graph/generated/node_generated.h>
#include <graph/generated/graph_generated.h>

#ifndef __JAVACPP_HACK__

namespace std {

    template <>
    class SD_EXPORT hash<std::pair<int, int>> {
    public:
        size_t operator()(const std::pair<int,int>& k) const;
    };

    template <>
    class SD_EXPORT hash<bfloat16> {
    public:
        size_t operator()(const bfloat16& k) const;
    };

    template <>
    class SD_EXPORT hash<float16> {
    public:
        size_t operator()(const float16& k) const;
    };
};

#endif

namespace sd {
    namespace graph {
        class SD_EXPORT Variable {
        protected:
            int _id = 0;
            int _index = 0;
            sd::NDArray *_ndarray = nullptr;
            std::string _name;

            std::vector<Nd4jLong> _shape;

            bool _external = false;
            bool _readOnly = false;
            bool _placeholder = false;
            bool _removable = true;

            // for now we're setting default to numeric
            // in future we'll be fetching it right from the array, 
            //InputType _variableType = InputType_UNDEFINED;
            //DataType _dataType = INHERIT;

            sd::NDArrayList *_list = nullptr;

            VariableType _variableType = VariableType::NDARRAY;
            
        public:
            Variable(bool placeHolder);
            Variable(sd::NDArray *arrayw, const char *name, int id, int idx = 0);
            Variable(sd::NDArray *array = nullptr, const char *name = nullptr);

#ifndef __JAVACPP_HACK__
            Variable(const sd::graph::FlatVariable *flatVariable);
#endif

            ~Variable();

            Variable* clone();

            template <typename N>
            SD_EXPORT Variable* asT();

            bool hasNDArray();
            sd::NDArray* getNDArray();
            void setNDArray(sd::NDArray *array);

            bool hasNDArrayList();
            sd::NDArrayList* getNDArrayList();
            void setNDArrayList(sd::NDArrayList* list);

            bool isExternal();
            bool isReadOnly();
            bool isEmpty();
            bool isRemovable();

            bool isPlaceholder();

            VariableType variableType();
            void setVariableType(VariableType variableType);

            /**
             * This method returns InputType of this variable  
             */
            //InputType variableType() {
            //    return _variableType;
            //}

            void markExternal(bool reallyExternal);
            void markReadOnly(bool reallyReadOnly);
            void markRemovable(bool reallyRemovable);

            int id();
            int index();
            void setIndex(int index);
            void setId(int id);
            void setId(int id, int idx);

            std::string *getName();
            void setName(std::string *name);

            std::vector<Nd4jLong>& shape();

#ifndef __JAVACPP_HACK__
            /**
             * This method returns offset to this Variable in FlatBuffer
             * @param builder
             * @return
             */
            flatbuffers::Offset<FlatVariable> asFlatVariable(flatbuffers::FlatBufferBuilder &builder);
#endif
        };
    }
}


#endif //LIBND4J_VARIABLE_H
