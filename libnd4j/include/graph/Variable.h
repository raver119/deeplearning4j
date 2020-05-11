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

#include <array/NDArray.h>
#include <array/NDArrayList.h>
#include <graph/VariableType.h>
#include <graph/generated/array_generated.h>
#include <graph/generated/graph_generated.h>
#include <graph/generated/node_generated.h>

#include <string>

#ifndef __JAVACPP_HACK__

namespace std {

template <>
class SD_EXPORT hash<std::pair<int, int>> {
 public:
  size_t operator()(const std::pair<int, int> &k) const;
};

template <>
class SD_EXPORT hash<bfloat16> {
 public:
  size_t operator()(const bfloat16 &k) const;
};

template <>
class SD_EXPORT hash<float16> {
 public:
  size_t operator()(const float16 &k) const;
};
};  // namespace std

#endif

namespace sd {
namespace graph {
class SD_EXPORT Variable {
 protected:
  int _id = 0;
  int _index = 0;
  std::shared_ptr<sd::NDArray> _ndarray;
  std::string _name;

  std::vector<Nd4jLong> _shape;
  DataType _dtype;

  bool _external = false;
  bool _readOnly = false;
  bool _placeholder = false;
  bool _removable = true;

  std::shared_ptr<sd::NDArrayList> _list;

  VariableType _variableType = VariableType::NDARRAY;

 public:
  explicit Variable(bool placeHolder, DataType dataType = DataType::ANY,
                    const std::vector<Nd4jLong> &shape = {});
  explicit Variable(const sd::NDArray &array, const std::string &name, int id,
                    int idx = 0);
  explicit Variable(std::shared_ptr<sd::NDArray> array, const std::string &name,
                    int id, int idx = 0);
  explicit Variable(std::shared_ptr<sd::NDArray> array,
                    const char *name = nullptr);
  explicit Variable(const NDArrayList &arrayList, const std::string &name,
                    int id, int idx = 0);
  explicit Variable();

#ifndef __JAVACPP_HACK__
  explicit Variable(const sd::graph::FlatVariable *flatVariable);
#endif

  ~Variable();

  bool hasNDArray() const;
  std::shared_ptr<sd::NDArray> getNDArray() const;
  void setNDArray(std::shared_ptr<sd::NDArray> array);

  bool hasNDArrayList() const;
  std::shared_ptr<sd::NDArrayList> getNDArrayList() const;
  void setNDArrayList(std::shared_ptr<sd::NDArrayList> list);

  bool isExternal() const;
  bool isReadOnly() const;
  bool isEmpty() const;
  bool isRemovable() const;

  bool isPlaceholder() const;

  VariableType variableType() const;
  void setVariableType(VariableType variableType);

  void markExternal(bool reallyExternal);
  void markReadOnly(bool reallyReadOnly);
  void markRemovable(bool reallyRemovable);

  int id() const;
  int index() const;
  void setIndex(int index);
  void setId(int id);
  void setId(int id, int idx);

  const std::string &name() const;
  const std::string &getName() const;
  void setName(const std::string &name);

  const std::vector<Nd4jLong> &shape() const;
  DataType dataType() const;

#ifndef __JAVACPP_HACK__
  /**
   * This method returns offset to this Variable in FlatBuffer
   * @param builder
   * @return
   */
  flatbuffers::Offset<FlatVariable> asFlatVariable(
      flatbuffers::FlatBufferBuilder &builder);
#endif
};
}  // namespace graph
}  // namespace sd

#endif  // LIBND4J_VARIABLE_H
