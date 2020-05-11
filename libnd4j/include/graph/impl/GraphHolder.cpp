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
//  @author raver119@gmail.com
//

#include <exceptions/graph_execution_exception.h>
#include <exceptions/graph_exists_exception.h>
#include <graph/GraphHolder.h>

namespace sd {
namespace graph {
GraphHolder* GraphHolder::getInstance() {
  if (_INSTANCE == nullptr) _INSTANCE = new GraphHolder();

  return _INSTANCE;
};

void GraphHolder::registerGraph(Nd4jLong graphId, const Graph& graph) {
  if (hasGraph(graphId)) throw graph_exists_exception(graphId);

  std::lock_guard<std::mutex> lock(_mutex);
  _graphs[graphId] = graph;
}

Graph& GraphHolder::graph(Nd4jLong graphId) {
  if (!this->hasGraph(graphId)) {
    nd4j_printf("GraphHolder doesn't have graph stored for [%lld]\n", graphId);
    throw std::runtime_error("Bad argument");
  }

  std::lock_guard<std::mutex> lock(_mutex);
  return _graphs[graphId];
}

void GraphHolder::forgetGraph(Nd4jLong graphId) {
  if (this->hasGraph(graphId)) {
    std::lock_guard<std::mutex> lock(_mutex);
    _graphs.erase(graphId);
  }
}

void GraphHolder::dropGraph(Nd4jLong graphId) { forgetGraph(graphId); }

bool GraphHolder::hasGraph(Nd4jLong graphId) {
  std::lock_guard<std::mutex> lock(_mutex);
  return _graphs.count(graphId) > 0;
}

void GraphHolder::replaceGraph(Nd4jLong graphId, const Graph& graph) {
  if (!hasGraph(graphId)) {
    registerGraph(graphId, graph);
    return;
  }

  forgetGraph(graphId);

  std::lock_guard<std::mutex> lock(_mutex);
  _graphs[graphId] = graph;
}

flatbuffers::Offset<FlatResult> GraphHolder::execute(
    Nd4jLong graphId, flatbuffers::FlatBufferBuilder& builder,
    const FlatInferenceRequest* request) {
  if (!hasGraph(graphId)) throw unknown_graph_exception(graphId);
  /*
              lockRead(graphId);

              auto graph = cloneGraph(graphId);
              auto res = GraphExecutioner::execute(graph, builder, request);
              delete graph;

              unlockRead(graphId);

              return res;
              */
  throw std::runtime_error("GraphHolder::execute - not implemented yet");
}

GraphHolder* GraphHolder::_INSTANCE = 0;
}  // namespace graph
}  // namespace sd
