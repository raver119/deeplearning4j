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

#ifndef LIBND4J_GNODE_H
#define LIBND4J_GNODE_H

#include <array/NDArray.h>
#include <graph/generated/node_generated.h>
#include <ops/declarable/DeclarableOp.h>
#include <system/pointercast.h>

#include <atomic>
#include <string>

#include "Context.h"

namespace sd {
namespace graph {

class Graph;

class SD_EXPORT Node {
 protected:
  // int and string IDs
  int _id = 0;
  std::string _name;

  // Node state, basically
  ContextPrototype _protoContext;

  // these 2 fields are used for Logic ops only
  OpType _opType;
  OpClass _opClass;
  Nd4jLong _opNum;

  // Inputs are stored in <Producer Node ID: Producer Node output index> format
  std::vector<std::pair<int, int>> _input;

  // Outputs are stored in <Producer Node output index: Consumer Node ID> format
  std::vector<std::pair<int, int>> _output;

  // Control flow dependencies for Node
  std::vector<std::pair<int, int>> _dependencies;
  std::vector<std::string> _stringDependencies;

  // service state fields
  bool _hasExternalOutputs = false;
  bool _hasExternalInputs = false;
  bool _hasInternalOutputs = false;
  bool _hasInternalInputs = false;

  std::shared_ptr<sd::ops::DeclarableOp> _customOp;

  // each node can be active or inactive, if used with divergents, like IF
  // statements
  bool _active = true;

 public:
  explicit Node(const sd::ops::DeclarableOp &op,
                const std::string &nodeName = {},
                const std::vector<double> &tArgs = {},
                const std::vector<Nd4jLong> &iArgs = {},
                const std::vector<bool> &bArgs = {},
                const std::vector<DataType> &dArgs = {});
  explicit Node(const std::string &opName, const std::string &nodeName = {},
                const std::vector<double> &tArgs = {},
                const std::vector<Nd4jLong> &iArgs = {},
                const std::vector<bool> &bArgs = {},
                const std::vector<DataType> &dArgs = {});
  explicit Node(const FlatNode *node);
  ~Node();

  /*
   * FIXME: deprecated methods, to be removed
   */
  explicit Node(const std::string &opName, const std::string &nodeName,
                const int id, const std::vector<std::string> &inputs = {},
                const std::vector<double> &tArgs = {},
                const std::vector<Nd4jLong> &iArgs = {});
  explicit Node(const std::string &opName, const int id = 0,
                const std::vector<std::pair<int, int>> &inputs = {},
                const std::vector<double> &tArgs = {},
                const std::vector<Nd4jLong> &iArgs = {});
  explicit Node(sd::ops::DeclarableOp *customOp, int id = 0,
                std::initializer_list<int> input = {},
                std::initializer_list<int> output = {},
                std::initializer_list<int> dimensions = {}, float scalar = 0.0f,
                std::initializer_list<double> tArgs = {},
                std::initializer_list<int> iArgs = {});
  explicit Node(std::shared_ptr<sd::ops::DeclarableOp> customOp, int id = 0,
                std::initializer_list<int> input = {},
                std::initializer_list<int> output = {},
                std::initializer_list<int> dimensions = {}, float scalar = 0.0f,
                std::initializer_list<double> tArgs = {},
                std::initializer_list<int> iArgs = {});
  explicit Node(OpType opType = OpType_TRANSFORM_SAME, int opNum = 0,
                int id = 0, std::initializer_list<int> input = {},
                std::initializer_list<int> output = {},
                std::initializer_list<int> dimensions = {}, float scalar = 0.0f,
                std::initializer_list<double> tArgs = {},
                std::initializer_list<int> iArgs = {});

  Node(const Node &other) noexcept;

  Node &operator=(const Node &other) noexcept;

  // move constructor
  Node(Node &&other) noexcept;

  // move assignment operator
  Node &operator=(Node &&other) noexcept;

  bool equals(const Node *other) const;
  bool equals(const Node &other) const;

  const ContextPrototype &protoContext() const;

  OpType opType() const { return _opType; };
  OpClass opClass() const { return _opClass;};

  Nd4jLong opNum() const;
  int id() const;

  const std::vector<std::pair<int, int>> &input() const;
  const std::vector<std::pair<int, int>> &output() const;
  const std::vector<std::pair<int, int>> &dependencies() const;

  void setId(int id);

  bool isMultiInput();
  bool isMultiOutput();

  bool isDivergencePoint();

  bool hasExternalOutputs() const;
  bool hasExternalInputs() const;
  bool hasInternalOutputs() const;
  bool hasInternalInputs() const;

  void pickOutputOnce(int outputId);
  void pickOutput(int outputId);
  void pickOutput(int nodeId, int outputId);

  void pickExternalOutput(int outputId);

  void pickInput(int inputId);
  void pickInput(int nodeId, int outputId);
  void pickInput(const std::pair<int, int> &id);
  void pickInput(const std::string &id);

  void setName(const std::string &name);
  const std::string &name() const;

  void setContextPrototype(const ContextPrototype &block);

  void setCustomOp(const std::shared_ptr<sd::ops::DeclarableOp> &customOp);
  std::shared_ptr<sd::ops::DeclarableOp> customOp() const;

  bool hasCustomOp() const;

  void setOpType(OpType opType);

  // this method converts string deps to int deps
  void actualizeDependencies(const MAP_IMPL<std::string, int> &lookupTable) const;

  // utility method that generates legacy ops out of OpType and OpNum
  static std::shared_ptr<sd::ops::DeclarableOp> buildOpByType(OpType opType, int numInputs, int numIArgs, int numTArgs, int opNum);
};
}  // namespace graph
}  // namespace sd

#endif  // LIBND4J_GNODE_H
