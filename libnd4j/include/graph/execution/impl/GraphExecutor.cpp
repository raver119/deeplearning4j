/*******************************************************************************
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

#include <graph/Graph.h>
#include <graph/execution/GraphExecutor.h>


namespace sd {
namespace graph {
Context GraphExecutor::prepareContext(
    const ContextPrototype &contextPrototype, VariableProxy &variableProxy,
    const GraphMemoryManager &memoryManager) const {
  // TODO: maybe we'll want to do something here?
  return Context(contextPrototype, &variableProxy,
                 const_cast<GraphMemoryManager *>(&memoryManager));
}

Nd4jStatus GraphExecutor::preprocess(sd::ops::DeclarableOp *op,
                                     Context &context) const {
  // time to allocate outputs, if that's not inplace op
  // inplace case is covered there
  op->prepareOutputs(context);

  // once prepareOutputs method was called - we don't need shape function
  // anymore
  context.setShapeFunctionOverride(true);

  return Status::OK();
}

Nd4jStatus GraphExecutor::postprocess(sd::ops::DeclarableOp *op,
                                      Context *context) const {
  return Status::OK();
}

Nd4jStatus GraphExecutor::execute(
    const std::shared_ptr<sd::ops::DeclarableOp> &op,
    const ContextPrototype &contextPrototype, const OpSequence &sequence,
    const OptimizedGraph &graph, VariableProxy &proxy,
    const int deviceId) const {
  auto ctx = prepareContext(contextPrototype, proxy,  GraphMemoryManager()/*graph.memoryManager()*/);
  return op->execute(&ctx);
  // throw std::runtime_error("GraphExecutor::execute - Not implemented yet");
}

Nd4jStatus GraphExecutor::execute(const OpSequence &seq,
                                  const OptimizedGraph &graph,
                                  std::deque<StackFrame> &stackFrames,
                                  const int deviceId) const {
  // we either follow or override target deviceId specified in OpSequence
  auto targetDevice = deviceId >= 0
      ? deviceId
      : seq.deviceId();

  /*
   * this is a basic implementation that works without dispatching etc
   */
  auto result = Status::OK();
  for (int e = 0; e < seq.length(); e++) {
    auto &v = seq[e];
    auto &p = stackFrames.back().variableProxy();

    // only Ops can be executed this way :(
    if (v.node().hasCustomOp())
      result = execute(v.node().customOp(), v.protoContext(), seq, graph, const_cast<VariableProxy&>(p), targetDevice);
    else {
      nd4j_printf("Node <%i:%s> has no customOp set\n",
                  v.node().id(),
                  v.node().name().empty() ? "" : v.node().name().c_str());
    }

    // if any one op fails - there will be no sense in executing other ops
    if (result != Status::OK()) return result;
  }

  return Status::OK();
}

Nd4jStatus GraphExecutor::execute(const OptimizedGraph &graph,
                                  VariableProxy &proxy,
                                  bool isInference) const {
  /*
   * this is a basic exection logic: roll through layers and sequences and
   * execute them one by one sequentially
   */
  std::deque<StackFrame> stackFrames;

  StackFrame baseFrame(proxy);

  // now we create one default StackFrame. current one.
  stackFrames.push_back(baseFrame);

  const auto numDevices = AffinityManager::numberOfDevices();
  Nd4jStatus result = Status::OK();  //
  for (uint64_t l = 0; l < graph.layers(); l++) {
    const auto &layer = graph.layer(l);

    //TODO: this loop is executable in parallel, so we should do this eventually
    for (uint64_t o = 0; o < layer.width(); o++) {
      result = execute(layer[o], graph, stackFrames, -1);
    }

    // early termination
    if (result != Status::OK()) return result;

    // optionally block until all sequences in this layer processed
    if (layer.width() > 0 && numDevices > 1)
      for (uint64_t o = 0; o < layer.width(); o++) {
        result = layer[o].wait();

        // early termination
        if (result != Status::OK()) return result;
      }
  }

  // that's the rule. it can't be not equal to 1.
  assert(stackFrames.size() == 1);

  return result;
}

}  // namespace graph
}  // namespace sd
