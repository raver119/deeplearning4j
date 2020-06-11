/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 * Copyright (c) 2020 Konduit K.K.
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

#include <array/NDArrayFactory.h>
#include <graph/FlatUtils.h>
#include <graph/Node.h>
#include <ops/declarable/LegacyBroadcastBoolOp.h>
#include <ops/declarable/LegacyBroadcastOp.h>
#include <ops/declarable/LegacyIndexReduceOp.h>
#include <ops/declarable/LegacyOp.h>
#include <ops/declarable/LegacyPairwiseTransformBoolOp.h>
#include <ops/declarable/LegacyPairwiseTransformOp.h>
#include <ops/declarable/LegacyRandomOp.h>
#include <ops/declarable/LegacyReduce3Op.h>
#include <ops/declarable/LegacyReduceBoolOp.h>
#include <ops/declarable/LegacyReduceFloatOp.h>
#include <ops/declarable/LegacyReduceLongOp.h>
#include <ops/declarable/LegacyReduceSameOp.h>
#include <ops/declarable/LegacyScalarBoolOp.h>
#include <ops/declarable/LegacyScalarOp.h>
#include <ops/declarable/LegacyStatsOp.h>
#include <ops/declarable/LegacyTransformBoolOp.h>
#include <ops/declarable/LegacyTransformFloatOp.h>
#include <ops/declarable/LegacyTransformSameOp.h>
#include <ops/declarable/LegacyTransformStrictOp.h>
#include <ops/declarable/OpRegistrator.h>
#include <helpers/StringUtils.h>

namespace sd {
namespace graph {
Node::Node(const ops::DeclarableOp &opName, const std::string &nodeName,
           const std::vector<double> &tArgs, const std::vector<Nd4jLong> &iArgs,
           const std::vector<bool> &bArgs, const std::vector<DataType> &dArgs) {
  auto customOp =
      ops::OpRegistrator::getInstance().getOperation(opName.getOpHash());

  this->_name = nodeName;
  this->_opType = OpType_CUSTOM;
  this->_opNum = customOp->getOpHash();
  this->_customOp = customOp;

  _hasExternalInputs = false;
  _hasExternalOutputs = false;
  _hasInternalInputs = false;
  _hasInternalOutputs = false;

  ContextPrototype block(this->customOp()->getOpDescriptor(), this->id(),
                         false);
  block.setName(nodeName);

  block.appendI(iArgs);
  block.appendT(tArgs);
  block.appendB(bArgs);
  block.appendD(dArgs);

  this->setContextPrototype(block);
}

Node::Node(const std::string &opName, const std::string &nodeName,
           const std::vector<double> &tArgs, const std::vector<Nd4jLong> &iArgs,
           const std::vector<bool> &bArgs, const std::vector<DataType> &dArgs) {
  auto customOp = ops::OpRegistrator::getInstance().getOperation(opName);

  this->_name = nodeName;
  this->_opType = OpType_CUSTOM;
  this->_opNum = customOp->getOpHash();
  this->_customOp = customOp;

  _hasExternalInputs = false;
  _hasExternalOutputs = false;
  _hasInternalInputs = false;
  _hasInternalOutputs = false;

  ContextPrototype block(this->customOp()->getOpDescriptor(), this->id(),
                         false);
  block.setName(nodeName);

  block.appendI(iArgs);
  block.appendT(tArgs);
  block.appendB(bArgs);
  block.appendD(dArgs);

  this->setContextPrototype(block);
}

bool Node::isDivergencePoint() {
  if (hasCustomOp()) {
    return _customOp->getOpDescriptor()->isDivergent();
  } else if (opType() == OpType_LOGIC && opNum() == 30)
    return true;
  else
    return false;
}

void Node::setContextPrototype(const ContextPrototype &block) {
  _protoContext = block;
}

void Node::setId(int id) {
  _id = id;
  _protoContext.setNodeId(id);
}

std::shared_ptr<sd::ops::DeclarableOp> Node::customOp() const {
  return _customOp;
}

void Node::setCustomOp(const std::shared_ptr<sd::ops::DeclarableOp>& customOp) {
  _customOp = customOp;
}

bool Node::hasCustomOp() const { return _customOp != nullptr; }

const std::string &Node::name() const { return _name; }


void Node::setName(const std::string &name) { _name = name; }

void Node::pickInput(const std::pair<int, int> &pair) {
  _input.push_back(pair);
  _protoContext.pickInput(pair);
}

void Node::pickInput(const std::string &id) {
  throw std::runtime_error("Node::pickInput - Not implemented yet");
}

void Node::pickInput(int inputId, int outputId) {
  std::pair<int, int> p(inputId, outputId);
  pickInput(p);
}

void Node::pickInput(int inputId) {
  pickInput(inputId, 0);

  if (inputId < 0)
    _hasExternalInputs = true;
  else
    _hasInternalInputs = true;
}

void Node::pickExternalOutput(int outputId) {
  std::pair<int, int> pair(outputId, 0);
  _output.push_back(pair);

  _hasExternalOutputs = true;
}

void Node::pickOutputOnce(int outputId) {
  std::pair<int, int> pair(outputId, 0);
  if (std::find(_output.begin(), _output.end(), pair) == _output.end())
    pickOutput(outputId);
}

void Node::pickOutput(int nodeId, int outputId) {
  std::pair<int, int> pair(nodeId, outputId);
  _output.emplace_back(pair);
}

void Node::pickOutput(int outputId) {
  std::pair<int, int> pair(outputId, 0);
  _output.emplace_back(pair);

  if (outputId < 0)
    _hasExternalOutputs = true;
  else
    _hasInternalOutputs = true;
}

bool Node::hasExternalOutputs() const { return _hasExternalOutputs; }

bool Node::hasExternalInputs() const { return _hasExternalInputs; }

bool Node::hasInternalOutputs() const { return _hasInternalOutputs; }

bool Node::hasInternalInputs() const { return _hasInternalInputs; }

bool Node::isMultiInput() { return _input.size() > 1; }

bool Node::isMultiOutput() { return _output.size() > 1; }

int Node::id() const { return _id; }

Nd4jLong Node::opNum() const { return _opNum; }

const std::vector<std::pair<int, int>> &Node::inputs() const { return _input; }

const std::vector<std::pair<int, int>> &Node::outputs() const { return _output; }

Node::Node(const std::string &opName, const std::string &nodeName, const int id,
           const std::vector<std::string> &inputs,
           const std::vector<double> &tArgs,
           const std::vector<Nd4jLong> &iArgs) {
  auto customOp = ops::OpRegistrator::getInstance().getOperation(opName);

  this->_opType = OpType_CUSTOM;
  this->_id = id;
  this->_opNum = customOp->getOpHash();
  this->_customOp = customOp;

  for (auto i : inputs) pickInput(i);

  ContextPrototype block(this->customOp()->getOpDescriptor(), this->id(),
                         false);

  block.appendI(iArgs);
  block.appendT(tArgs);

  this->setContextPrototype(block);
}

Node::Node(const std::string &opName, const int id,
           const std::vector<std::pair<int, int>> &inputs,
           const std::vector<double> &tArgs,
           const std::vector<Nd4jLong> &iArgs) {
  auto customOp = ops::OpRegistrator::getInstance().getOperation(opName);

  this->_opType = OpType_CUSTOM;
  this->_id = id;
  this->_opNum = customOp->getOpHash();
  this->_customOp = customOp;


  for (auto i : inputs) pickInput(i);

  ContextPrototype block(this->customOp()->getOpDescriptor(), this->id(),
                         false);

  block.appendI(iArgs);
  block.appendT(tArgs);

  this->setContextPrototype(block);
}

Node::Node(sd::ops::DeclarableOp *customOp, int id,
           std::initializer_list<int> input, std::initializer_list<int> output,
           std::initializer_list<int> dimensions, float scalar,
           std::initializer_list<double> tArgs,
           std::initializer_list<int> iArgs) {
  this->_opType = OpType_CUSTOM;
  this->_id = id;
  this->_opNum = customOp->getOpHash();

  // if custom op is a registered one - pull it from cache, otherwise - clone
  // locally
  if (sd::ops::OpRegistrator::getInstance().hasOperation(_opNum))
    this->_customOp =
        sd::ops::OpRegistrator::getInstance().getOperation(_opNum);
  else
    throw std::runtime_error(
        "Can't create a node with custom operation within");

  for (auto i : input) pickInput(i);

  for (auto o : output) pickOutput(o);

  ContextPrototype block(this->customOp()->getOpDescriptor(), this->id(),
                         false);

  for (auto v : dimensions) block.appendA(v);

  for (auto v : iArgs) block.appendI(v);

  for (auto v : tArgs) block.appendT(v);

  this->setContextPrototype(block);
}

const std::vector<std::pair<int, int>>& Node::dependencies() const {
  return _dependencies;
}

void Node::actualizeDependencies(const MAP_IMPL<std::string, int> &lookupTable) const {
  for (const auto &v: _stringDependencies) {
    if (lookupTable.count(v) == 0)
      throw std::runtime_error("Unknown Node dependency found: [" + v + "]");

    const_cast<Node*>(this)->_dependencies.emplace_back(std::pair<int, int>{lookupTable.at(v), 0});
  }
}

Node::Node(OpType opType, int opNum, int id, std::initializer_list<int> input,
           std::initializer_list<int> output,
           std::initializer_list<int> dimensions, float scalar,
           std::initializer_list<double> tArgs,
           std::initializer_list<int> iArgs) {
  this->_opType = opType;
  this->_id = id;
  this->_opNum = opNum;

  _hasExternalInputs = false;
  _hasExternalOutputs = false;
  _hasInternalInputs = false;
  _hasInternalOutputs = false;

  for (auto i : input) pickInput(i);

  for (auto o : output) pickOutput(o);

  // these ops allow in-place execution by design
  if (opType == OpType_TRANSFORM_SAME || opType == OpType_TRANSFORM_FLOAT ||
      opType == OpType_TRANSFORM_STRICT || opType == OpType_TRANSFORM_BOOL ||
      opType == OpType_SCALAR || opType == OpType_BROADCAST) {

    _opClass = OpClass_TRANSFORM;
  } else if (opType == OpType_REDUCE_SAME || opType == OpType_REDUCE_FLOAT ||
             opType == OpType_REDUCE_BOOL || opType == OpType_REDUCE_LONG ||
             opType == OpType_SUMMARYSTATS) {
    _opClass = OpClass_REDUCTION;
  }

  if (opType == OpType_BROADCAST || opType == OpType_BROADCAST_BOOL ||
      opType == OpType_INDEX_REDUCE || opType == OpType_SUMMARYSTATS ||
      opType == OpType_REDUCE_BOOL || opType == OpType_REDUCE_SAME ||
      opType == OpType_REDUCE_FLOAT || opType == OpType_REDUCE_3 ||
      opType == OpType_TRANSFORM_STRICT || opType == OpType_TRANSFORM_SAME ||
      opType == OpType_TRANSFORM_FLOAT || opType == OpType_TRANSFORM_BOOL ||
      opType == OpType_RANDOM || opType == OpType_PAIRWISE ||
      opType == OpType_PAIRWISE_BOOL || opType == OpType_SCALAR_BOOL ||
      opType == OpType_SCALAR) {
    ContextPrototype block(nullptr, this->id(), false);

    for (auto v : dimensions) block.appendA(v);

    for (auto v : iArgs) block.appendI(v);

    for (auto v : tArgs) block.appendT(v);

    this->setContextPrototype(block);

    this->setCustomOp(Node::buildOpByType(
        opType, (int)input.size(), (int)block.getIArguments().size(),
        (int)block.getTArguments().size(), opNum));
    block.setOpDescriptor(this->customOp()->getOpDescriptor());
  } else if (opType == OpType_CUSTOM) {
    if (this->customOp()) {
      ContextPrototype block(this->customOp()->getOpDescriptor(), this->id(),
                             false);

      for (auto v : dimensions) block.appendA(v);

      for (auto v : iArgs) block.appendI(v);

      for (auto v : tArgs) block.appendT(v);

      this->setContextPrototype(block);
    } else
      throw std::runtime_error("wrong custom operation given");
  }
};

Node::Node(const FlatNode *node) {
  // temporary holders _dimensions, for transferring axis into ContextPrototype
  std::vector<int> axis;

  if (node->scalar() != nullptr)
    throw std::runtime_error("FlatNode has scalar defined, it's deprecated");

  if (node != nullptr) {
    this->_id = node->id();
    // this->_dataType = DataTypeUtils::fromFlatDataType(node->dataType());
    this->_opNum = node->opNum();
    this->_opType = node->opType();

    if (node->name() != nullptr && node->name()->c_str() != nullptr) {
      this->_name = node->name()->str();
    }

    if (node->inputPaired() != nullptr && node->inputPaired()->size() > 0) {
      for (int e = 0; e < (int)node->inputPaired()->size(); e++) {
        auto pair = node->inputPaired()->Get(e);
        pickInput(pair->first(), pair->second());
      }
    } else if (node->input() != nullptr && node->input()->size() > 0) {
      for (int e = 0; e < (int)node->input()->size(); e++)
        pickInput(node->input()->Get(e));
    } else {
      if (this->opType() != OpType_LOGIC) {
        if (this->_name.size() > 0) {
          nd4j_debug("Node [%i:<%s>] has no inputs defined\n", this->_id, this->_name.c_str());
        } else {
          nd4j_debug("Node [%i:<noname>] has no inputs defined\n", this->_id);
        }
      }
    }

    // reading control deps, and filling _dependencies field
    if (node->varControlDeps() != nullptr && node->varControlDeps()->size() > 0)
      for (int e = 0; e < node->varControlDeps()->size(); e++)
        _stringDependencies.emplace_back(node->varControlDeps()->Get(e)->str());

    if (node->controlDepFor() != nullptr && node->controlDepFor()->size() > 0)
      for (int e = 0; e < node->controlDepFor()->size(); e++)
        _stringDependencies.emplace_back(node->controlDepFor()->Get(e)->str());

    if (node->controlDeps() != nullptr && node->controlDeps()->size() > 0)
      for (int e = 0; e < node->controlDeps()->size(); e++)
        _stringDependencies.emplace_back(node->controlDeps()->Get(e)->str());


    // transferring dimensions. Used for legacy ops only
    if (node->dimensions() != nullptr && node->dimensions()->size() > 0) {
      axis.resize(node->dimensions()->size());

      for (int e = 0; e < (int) node->dimensions()->size(); e++)
        axis[e] = node->dimensions()->Get(e);
    }

    if (this->opType() == OpType_LOGIC && this->opNum() == 100L) {
      if (node->extraInteger()->size() < 1)
        throw std::runtime_error("Enter Node [" + StringUtils::valueToString(this->id()) + "] must have FrameID specified");

      //this->setFrameId(node->extraInteger()->Get(0));
    }

    // these ops allow in-place execution by design
    if (_opType == OpType_BROADCAST || _opType == OpType_BROADCAST_BOOL ||
        _opType == OpType_INDEX_REDUCE || _opType == OpType_SUMMARYSTATS ||
        _opType == OpType_REDUCE_BOOL || _opType == OpType_REDUCE_SAME ||
        _opType == OpType_REDUCE_FLOAT || _opType == OpType_REDUCE_3 ||
        _opType == OpType_TRANSFORM_STRICT ||
        _opType == OpType_TRANSFORM_SAME || _opType == OpType_TRANSFORM_FLOAT ||
        _opType == OpType_TRANSFORM_BOOL || _opType == OpType_RANDOM ||
        _opType == OpType_PAIRWISE || _opType == OpType_PAIRWISE_BOOL ||
        _opType == OpType_SCALAR_BOOL || _opType == OpType_SCALAR) {

      if (node->input() != nullptr && node->input()->size() > 0) {
        ContextPrototype block(nullptr, this->id(), false);
        if (!this->name().empty())
          block.setName(this->name());

        for (auto v : axis) block.appendA(v);

        if (node->extraParams() != nullptr && node->extraParams()->size() > 0)
          for (int e = 0; e < (int) node->extraParams()->size(); e++)
            block.appendT(static_cast<double>(node->extraParams()->Get(e)));

        if (node->extraBools() != nullptr && node->extraBools()->size() > 0)
          for (int e = 0; e < (int) node->extraBools()->size(); e++)
            block.appendB(node->extraBools()->Get(e));

        if (node->extraInteger() != nullptr && node->extraInteger()->size() > 0)
          for (int e = 0; e < (int) node->extraInteger()->size(); e++)
            block.appendI(node->extraInteger()->Get(e));

        if (node->extraTypes() != nullptr && node->extraTypes()->size() > 0)
          for (int e = 0; e < (int) node->extraTypes()->size(); e++)
            block.appendD((sd::DataType) node->extraTypes()->Get(e));

        this->setContextPrototype(block);
        this->setCustomOp(Node::buildOpByType(
            _opType, (int) node->input()->size(),
            (int) block.getIArguments().size(),
            (int) block.getTArguments().size(), (int) _opNum));
        block.setOpDescriptor(this->customOp()->getOpDescriptor());
      } else if (node->inputPaired() != nullptr &&
          node->inputPaired()->size() > 0) {
        ContextPrototype block(nullptr, this->id(), false);

        for (int e = 0; e < this->inputs().size(); e++) {
          block.pickInput(this->inputs().at(e));
        }

        // there's no other IArgs in legacy options, actually
        for (auto v : axis) block.appendA(v);

        if (node->extraParams() != nullptr && node->extraParams()->size() > 0)
          for (int e = 0; e < (int) node->extraParams()->size(); e++)
            block.appendT(static_cast<double>(node->extraParams()->Get(e)));

        if (node->extraBools() != nullptr && node->extraBools()->size() > 0)
          for (int e = 0; e < (int) node->extraBools()->size(); e++)
            block.appendB(node->extraBools()->Get(e));

        if (node->extraInteger() != nullptr && node->extraInteger()->size() > 0)
          for (int e = 0; e < (int) node->extraInteger()->size(); e++)
            block.appendI(node->extraInteger()->Get(e));

        if (node->extraTypes() != nullptr && node->extraTypes()->size() > 0)
          for (int e = 0; e < (int) node->extraTypes()->size(); e++)
            block.appendD((sd::DataType) node->extraTypes()->Get(e));

        this->setContextPrototype(block);

        this->setCustomOp(Node::buildOpByType(
            _opType, (int) node->inputPaired()->size(),
            (int) block.getIArguments().size(),
            (int) block.getTArguments().size(), (int) _opNum));
        block.setOpDescriptor(this->customOp()->getOpDescriptor());
      }
    } else if (this->_opType == OpType_LOGIC) {
      ContextPrototype block(nullptr, this->id());
      if (!this->name().empty())
        block.setName(this->name());

      for (int e = 0; e < this->inputs().size(); e++)
        block.pickInput(this->inputs().at(e));

      this->setContextPrototype(block);
    } else if (this->_opType == OpType_CUSTOM) {
      auto op =
          sd::ops::OpRegistrator::getInstance().getOperation(this->opNum());
      if (op == nullptr)
        throw std::runtime_error("Can't find requested operation [" + StringUtils::valueToString(this->opNum()) + "]");

      ContextPrototype block(nullptr, this->id());
      if (!this->name().empty())
        block.setName(this->name());

      for (int e = 0; e < this->inputs().size(); e++)
        block.pickInput(this->inputs().at(e));

      if (node->extraInteger() != nullptr)
        for (uint32_t e = 0; e < node->extraInteger()->size(); e++) {
          auto v = node->extraInteger()->Get(e);
          // FIXME: remove this static_cast, iArgs should be Nd4jLong
          block.appendI(static_cast<int>(v));
        }

      if (node->extraParams() != nullptr)
        for (uint32_t e = 0; e < node->extraParams()->size(); e++)
          block.appendT(static_cast<double>(node->extraParams()->Get(e)));

      if (node->extraBools() != nullptr && node->extraBools()->size() > 0)
        for (int e = 0; e < (int)node->extraBools()->size(); e++) {
          block.appendB(node->extraBools()->Get(e));
        }

      if (node->extraTypes() != nullptr && node->extraTypes()->size() > 0) {
        for (int e = 0; e < (int)node->extraTypes()->size(); e++) {
          block.appendD((sd::DataType)node->extraTypes()->Get(e));
        }
      }

      for (auto v : axis) block.appendA(v);

      this->setContextPrototype(block);
      this->setCustomOp(op);
      block.setOpDescriptor(this->customOp()->getOpDescriptor());
    }
  } else {
    // empty dynamic node, tests probably
  }
}


const ContextPrototype &Node::contextPrototype() const { return _protoContext; }

Node::~Node() { }

bool Node::equals(const Node *other) const {
  if (_opType == other->_opType && _opNum == other->_opNum)
    return true;

  return false;
}

bool Node::equals(const Node &other) const {
  return this->equals(&other);
}

Node::Node(const Node &other) noexcept {
  _opType = other._opType;
  _opClass = other._opClass;
  _opNum = other._opNum;
  _customOp = other._customOp;
  _name = other._name;
  _id = other._id;

  _hasExternalOutputs = other._hasExternalOutputs;
  _hasExternalInputs = other._hasExternalInputs;
  _hasInternalOutputs = other._hasInternalOutputs;
  _hasInternalInputs = other._hasInternalInputs;
  _active = other._active;

  _customOp = other._customOp;
  _protoContext = other._protoContext;

  _input = other._input;
  _output = other._output;
}

Node &Node::operator=(const Node &other) noexcept {
  if (this == &other) return *this;

  _opType = other._opType;
  _opClass = other._opClass;
  _opNum = other._opNum;
  _customOp = other._customOp;
  _name = other._name;
  _id = other._id;

  _hasExternalOutputs = other._hasExternalOutputs;
  _hasExternalInputs = other._hasExternalInputs;
  _hasInternalOutputs = other._hasInternalOutputs;
  _hasInternalInputs = other._hasInternalInputs;
  _active = other._active;

  _customOp = other._customOp;
  _protoContext = other._protoContext;

  _input = other._input;
  _output = other._output;

  return *this;
}

Node::Node(Node &&other) noexcept {

  _opType = other._opType;
  _opClass = other._opClass;
  _opNum = other._opNum;
  _customOp = other._customOp;
  _name = std::move(other._name);
  _id = other._id;

  _hasExternalOutputs = other._hasExternalOutputs;
  _hasExternalInputs = other._hasExternalInputs;
  _hasInternalOutputs = other._hasInternalOutputs;
  _hasInternalInputs = other._hasInternalInputs;
  _active = other._active;

  _protoContext = std::move(other._protoContext);

  _customOp = std::move(other._customOp);
  _input = std::move(other._input);
  _output = std::move(other._output);
}

Node &Node::operator=(Node &&other) noexcept {
  if (this == &other) return *this;

  _opType = other._opType;
  _opClass = other._opClass;
  _opNum = other._opNum;
  _customOp = other._customOp;
  _name = std::move(other._name);
  _id = other._id;

  _hasExternalOutputs = other._hasExternalOutputs;
  _hasExternalInputs = other._hasExternalInputs;
  _hasInternalOutputs = other._hasInternalOutputs;
  _hasInternalInputs = other._hasInternalInputs;
  _active = other._active;

  _protoContext = std::move(other._protoContext);

  _customOp = std::move(other._customOp);
  _input = std::move(other._input);
  _output = std::move(other._output);

  return *this;
}

std::shared_ptr<sd::ops::DeclarableOp> Node::buildOpByType(
    OpType opType, int numInputs, int numIArgs, int numTArgs, int opNum) {
  switch (opType) {
    case OpType_PAIRWISE:
      return std::make_shared<sd::ops::LegacyPairwiseTransformOp>(opNum);
    case OpType_PAIRWISE_BOOL:
      return std::make_shared<sd::ops::LegacyPairwiseTransformBoolOp>(opNum);
    case OpType_TRANSFORM_STRICT:
      return std::make_shared<sd::ops::LegacyTransformStrictOp>(opNum);
    case OpType_TRANSFORM_SAME:
      return std::make_shared<sd::ops::LegacyTransformSameOp>(opNum);
    case OpType_TRANSFORM_FLOAT:
      return std::make_shared<sd::ops::LegacyTransformFloatOp>(opNum);
    case OpType_TRANSFORM_BOOL:
      return std::make_shared<sd::ops::LegacyTransformBoolOp>(opNum);
    case OpType_SCALAR:
      return std::make_shared<sd::ops::LegacyScalarOp>(opNum);
    case OpType_SCALAR_BOOL:
      return std::make_shared<sd::ops::LegacyScalarBoolOp>(opNum);
    case OpType_REDUCE_3:
      return std::make_shared<sd::ops::LegacyReduce3Op>(opNum);
    case OpType_REDUCE_SAME:
      return std::make_shared<sd::ops::LegacyReduceSameOp>(opNum);
    case OpType_REDUCE_FLOAT:
      return std::make_shared<sd::ops::LegacyReduceFloatOp>(opNum);
    case OpType_REDUCE_LONG:
      return std::make_shared<sd::ops::LegacyReduceLongOp>(opNum);
    case OpType_REDUCE_BOOL:
      return std::make_shared<sd::ops::LegacyReduceBoolOp>(opNum);
    case OpType_INDEX_REDUCE:
      return std::make_shared<sd::ops::LegacyIndexReduceOp>(opNum);
    case OpType_SUMMARYSTATS:
      return std::make_shared<sd::ops::LegacyStatsOp>(opNum);
    case OpType_RANDOM:
      return std::make_shared<sd::ops::LegacyRandomOp>(opNum);
    case OpType_BROADCAST:
      return std::make_shared<sd::ops::LegacyBroadcastOp>(opNum);
    case OpType_BROADCAST_BOOL:
      return std::make_shared<sd::ops::LegacyBroadcastBoolOp>(opNum);
    default:
      throw std::runtime_error("Bad opType passed in");
  }
}

}  // namespace graph
}  // namespace sd
