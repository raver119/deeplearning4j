/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 * Copyright (c) 2019-2020 Konduit K.K.
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
//  @author raver119@gmail.com
//

#ifndef ND4J_CONTEXT_PROTOTYPE_H
#define ND4J_CONTEXT_PROTOTYPE_H

#include <array/DataType.h>
#include <execution/Engine.h>
#include <execution/ExecutionMode.h>
#include <graph/RandomGenerator.h>
#include <ops/declarable/OpDescriptor.h>
#include <system/Environment.h>
#include <system/dll.h>

#include <vector>

#ifndef __STANDALONE_BUILD__
#include <config.h>
#endif

namespace sd {
namespace graph {

class SD_EXPORT ContextPrototype {
 protected:
  // int ids of the input nodes
  std::vector<std::pair<int, int>> _inputs;
  int _nodeId;
  std::string _name;
  std::vector<double> _tArgs;
  std::vector<int> _iArgs;
  std::vector<bool> _bArgs;
  std::vector<int> _axis;
  std::vector<sd::DataType> _dArgs;

  bool _isInplace;

  // opNum for legacy XYZ ops
  int _opNum = -1;
  uint64_t _rootSeed;
  RandomGenerator _randomGenerator;

  sd::ops::OpDescriptor* _opDescriptor;
  bool _useMKLDNN = sd::Environment::getInstance()->isUseMKLDNN();

  // target engine for execution
  samediff::Engine _engine = DEFAULT_ENGINE;

  samediff::ExecutionMode _execMode = samediff::ExecutionMode::MODE_UNDEFINED;

 public:
  explicit ContextPrototype(sd::ops::OpDescriptor* opDescriptor = nullptr,
                            int nodeId = 1, bool inPlace = false);
  ~ContextPrototype() = default;

  ContextPrototype(const ContextPrototype& other) noexcept;

  ContextPrototype& operator=(const ContextPrototype& other) noexcept;

  // move constructor
  ContextPrototype(ContextPrototype&& other) noexcept;

  // move assignment operator
  ContextPrototype& operator=(ContextPrototype&& other) noexcept;

  int getNodeId() const;
  int nodeId() const;
  void setNodeId(int id);

  // this method returns true, if inputs are defined
  bool hasVariablesFilled() const;

  void setOpDescriptor(sd::ops::OpDescriptor* opDescriptor);

  bool isInplace() const;
  void markInplace(bool reallyInplace);

  void pickInput(int input);
  void pickInput(int input, int index);
  void pickInput(const std::pair<int, int>& p);
  void fillInputs(std::initializer_list<int> inputs);
  void fillInputs(std::vector<int>& inputs);
  const std::vector<std::pair<int, int>>& inputs() const;

  const std::vector<double>& getTArguments() const;
  const std::vector<int>& getIArguments() const;
  const std::vector<bool>& getBArguments() const;
  const std::vector<sd::DataType>& getDArguments() const;
  const std::vector<int>& getAxis() const;

  void appendI(const std::vector<Nd4jLong>& value);
  void appendT(const std::vector<double>& value);
  void appendB(const std::vector<bool>& value);
  void appendD(const std::vector<DataType>& value);

  void appendA(Nd4jLong value);
  void appendI(Nd4jLong value);
  void appendT(double value);
  void appendB(bool value);
  void appendD(DataType value);

  samediff::Engine engine() const;

  size_t numT() const;
  size_t numI() const;
  size_t numB() const;
  size_t numD() const;

  const std::pair<int, int>& input(int idx) const;

  int opNum() const;
  void setOpNum(int opNum);

  bool isUseMKLDNN() const { return _useMKLDNN; }
  void setUseMKLDNN(bool useMKLDNN) { _useMKLDNN = useMKLDNN; }

  std::string name() const;
  void setName(const std::string& name);

  /**
   * This method returns number of inputs available in this block
   * @return
   */
  virtual unsigned long width() const;

  // just a clone
  ContextPrototype* clone();

  template <typename N>
  ContextPrototype* asT();

  RandomGenerator& randomGenerator() { return _randomGenerator; }
  RandomGenerator const& getRng() const { return _randomGenerator; }
  void setRng(RandomGenerator const& anotherRng) {
    _randomGenerator = anotherRng;
  }
  void setRandomGenerator(RandomGenerator const& anotherRng) {
    _randomGenerator = anotherRng;
  }
  uint64_t randomSeed() const { return _rootSeed; }
  void setRandomSeed(uint64_t seed) { _rootSeed = seed; }
};
}  // namespace graph
}  // namespace sd

#endif  // ND4J_CONTEXT_PROTOTYPE_H
