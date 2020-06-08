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

#include <array/DataTypeUtils.h>
#include <exceptions/datatype_exception.h>
#include <exceptions/graph_exception.h>
#include <exceptions/graph_execution_exception.h>
#include <exceptions/shape_mismatch_exception.h>
#include <graph/FlatUtils.h>
#include <graph/Graph.h>
#include <graph/VariableProxy.h>
#include <graph/exceptions/unresolved_input_exception.h>
#include <graph/exceptions/unresolved_output_exception.h>
#include <helpers/EnumUtils.h>
#include <helpers/FileUtils.h>
#include <helpers/ShapeUtils.h>
#include <legacy/NativeOps.h>
#include <ops/declarable/OpRegistrator.h>

#include <vector>

namespace sd {
namespace graph {
const std::vector<std::shared_ptr<Variable>> &Graph::placeholders() const {
  return _variableSpace.placeholders();
}

int Graph::numberOfPlaceholders() const {
  return _variableSpace.numberOfPlaceholders();
};

const ExecutorConfiguration &Graph::getExecutorConfiguration() const {
  return _configuration;
}

VariableSpace &Graph::variableSpace() const {
  return const_cast<VariableSpace &>(_variableSpace);
}

Graph::~Graph() {}

int Graph::idByName(const std::string &nodeName) const {
  if (_symbolicLookupTable.count(nodeName) == 0)
    throw std::runtime_error("Can't find node [" + nodeName + "]");

  return _symbolicLookupTable.at(nodeName);
}

void Graph::addVariable(const std::string &name, NDArray &array) {
  int id = _maxId++;
  _symbolicLookupTable[name] = id;
  _variableSpace.putVariable(id, 0, array);
}

void Graph::addVariable(const std::string &name, NDArray &&array) {
  auto lvalue = array;
  addVariable(name, lvalue);
}

void Graph::addNode(Node &&node,
                    const std::initializer_list<std::string> &inputs) {
  auto lvalue = std::move(node);
  addNode(lvalue, inputs);
}

void Graph::addNode(Node &node,
                    const std::initializer_list<std::string> &inputs) {
  // temporary check. basically we're okay if Node has id defined
  if (node.id() != 0)
    throw std::runtime_error("Graph::addNode - Node has id defined");

  if (node.name().empty()) {
    // if name is empty we'll make up a name based on Op name
  } else {
    if (_symbolicLookupTable.count(node.name()) > 0)
      throw std::runtime_error("Graph::addNode - Graph alread has Node [" +
                               node.name() + "] defined");
  }

  // node must have numeric id
  node.setId(_maxId++);
  _symbolicLookupTable[node.name()] = node.id();

  // converting string ids to numeric ones
  for (auto &v : inputs) {
    // we don't allow self-references
    if (v == node.name())
      throw unresolved_input_exception::build(
          "Graph::addNode - Node references itself", v);

    node.pickInput(idByName(v), 0);
  }

  // actually storing the node. Later, topological sort will be applied on this
  // map
  _unmapped[node.id()] = node;
}

void Graph::addNode(Node &node, const std::initializer_list<int> &inputs) {
  throw std::runtime_error("Graph::addNode() - Not implemented yet");
}

void Graph::addNode(Node &node,
                    const std::initializer_list<std::pair<int, int>> &inputs) {
  node.markRemovable(false);

  throw std::runtime_error("Graph::addNode() - Not implemented yet");
}

Graph::Graph(const FlatGraph *flatGraph, const GraphMemoryManager &memoryManager): _memoryManager(memoryManager) {
  bool trusted = flatGraph != nullptr;

  // if there was no exec configuration in flatgraph - create default one
  if (flatGraph != nullptr && flatGraph->configuration() != nullptr) {
    _configuration = ExecutorConfiguration(flatGraph->configuration());
  } else
    _configuration = ExecutorConfiguration();

  // if memory reqs were set - initialize workspace
  if (_configuration._footprintForward > 0) {
    _workspace.expandBy(_configuration._footprintForward);
  }

  // parsing variables here
  if (flatGraph != nullptr && flatGraph->variables() != nullptr &&
      flatGraph->variables()->size() > 0) {
    for (unsigned int e = 0; e < flatGraph->variables()->size(); e++) {
      auto flatVar = flatGraph->variables()->Get(e);
      std::pair<int, int> pair(flatVar->id()->first(), flatVar->id()->second());

      auto var = std::make_shared<Variable>(flatVar);
      if (flatVar->name() != nullptr) {
        var->setName(flatVar->name()->str());
        _symbolicLookupTable[var->name()] = pair.first;
      }

      _variableSpace.putVariable(pair, var);
    }
  }

  // at this point we expect all variables are already registered
  // we're saving outputs only if explicit mode is set
  if (_configuration._outputMode == OutputMode_EXPLICIT ||
      _configuration._outputMode == OutputMode_EXPLICIT_AND_IMPLICIT) {
    if (flatGraph != nullptr && flatGraph->outputs() != nullptr) {
      for (unsigned int e = 0; e < flatGraph->outputs()->size(); e++) {
        auto out = flatGraph->outputs()->Get(e);
        std::pair<int, int> vp(out->first(), out->second());
        if (!_variableSpace.hasVariable(vp)) {
          nd4j_verbose("Non-existent variable requested: %i\n", out);
          throw std::runtime_error("Non-existent variable requested");
        }
      }
    }
  }

  // rolling through nodes
  if (flatGraph != nullptr && flatGraph->nodes() != nullptr &&
      flatGraph->nodes()->size() > 0) {
    for (unsigned int e = 0; e < flatGraph->nodes()->size(); e++) {
      auto node = flatGraph->nodes()->Get(e);

      if (node->output() == nullptr || node->output()->size() == 0) {
        nd4j_verbose("Orphan node detected: %i; AutoOutput to be considered\n",
                     node->id());
      }

      nd4j_debug("Node name: [%s]\n", node->name()->c_str());
      Node nnode(node);
      // just filling list of nodes
      _unmapped[nnode.id()] = nnode;

      if (!nnode.name().empty())
        _symbolicLookupTable[nnode.name()] = nnode.id();
    }
  }

  // now, once everything is deserializerd, time to roll through Variables/Nodes and update dependencies
  for (const auto &v: _unmapped)
    v.second.actualizeDependencies(_symbolicLookupTable);

  for (const auto &v:_variableSpace.variables())
    v->actualizeDependencies(_symbolicLookupTable);
}

/**
 * This method returns total number of nodes in this graph
 * @return
 */
int Graph::size() const { return _unmapped.size(); }

Nd4jStatus Graph::validate() {
  throw std::runtime_error("Graph::validate - method not implemented");
};

void Graph::printOutNode(const Node &node) const {
  nd4j_printf("%i. ", node.id());
  switch (node.opType()) {
    case OpType_CUSTOM: {
      printf("%s; ", node.customOp()->getOpName().c_str());
    } break;
    case OpType_LOGIC: {
      printf("%s; ", EnumUtils::_LogicOpToString(node.opNum()));
    } break;
    default: {
      printf("%s:{%i}; ", EnumUtils::_OpTypeToString(node.opType()),
             (int)node.opNum());
    }
  }

  nd4j_printf("Inputs: [", "");
  // auto block = node->getBlock();
  for (int e = 0; e < node.input().size(); e++) {
    auto in = node.input()[e];
    printf("{%i:%i}", in.first, in.second);
    if (e < node.input().size() - 1) nd4j_printf(", ", "");
  }

  if (node.opType() == OpType_CUSTOM) {
    auto ctx = node.protoContext();
    if (ctx.numI() > 0) {
      printf("]; iArgs: [");

      for (int e = 0; e < ctx.numI(); e++) {
        printf("%i", ctx.getIArguments().at(e));
        if (e < ctx.getIArguments().size() - 1) nd4j_printf(", ", "");
      }
    }
  }

  nd4j_printf("]; \n", "");

  //            printf("\n");
  fflush(stdout);
}

void Graph::printOut() {
  // print variables first
  if (_variableSpace.totalEntries() > 0) {
    nd4j_printf("\nPrinting out Variables...\n", "");
    auto vars = _variableSpace.variables();

    for (auto &v : vars) {
      if (v->hasNDArray()) {
        auto shape = ShapeUtils::shapeAsString(v->getNDArray().get());
        auto values = v->getNDArray()->asString(16);
        auto dtype = DataTypeUtils::asString(v->getNDArray()->dataType());

        if (!v->getName().empty()) {
          nd4j_printf("<%s> <%i:%i> dtype: %s; shape: %s; values: %s;\n",
                      v->getName().c_str(), v->id(), v->index(), dtype.c_str(),
                      shape.c_str(), values.c_str());
        } else {
          nd4j_printf("<%i:%i> dtype: %s; shape: %s; values: %s;\n", v->id(),
                      v->index(), dtype.c_str(), shape.c_str(), values.c_str());
        }
      } else if (v->hasNDArrayList()) {
        // TODO: add better NDArrayList printout
        nd4j_printf("<%i:%i> holds ArrayList", v->id(), v->index());
      }
    }
  }

  fflush(stdout);

  if (size() > 0) {
    nd4j_printf("\nPrinting out Nodes...\n", "");
    optimizedGraph().printOut();
  }
}

Nd4jStatus Graph::validateNode(Node *node) {
  // TODO: to be implemented
  return ND4J_STATUS_OK;
}

void Graph::replaceState(VariableSpace *state,
                         const ExecutorConfiguration &configuration) {
  _variableSpace = *state;
  _configuration = configuration;
}

Graph Graph::cloneWithProxy() const {
  Graph clone;

  // clone.replaceState(new VariableProxy(&this->_variableSpace),
  // this->_configuration);

  // return clone;
  throw std::runtime_error("Graph::cloneWithProxy - Not implemented yet");
}

Graph *Graph::clone() const {
  auto clone = new Graph();

  // clone->replaceState(&this->_variableSpace, this->_configuration.clone());

  throw std::runtime_error("Graph::clone - not implemented yet");
}

Nd4jLong Graph::hashCode() const {
  throw std::runtime_error("Graph::hashCode - not implemented yet");
}

Graph Graph::fromFlatBuffers(const char *fileName,
                             const GraphMemoryManager &memoryManager) {
  // check if file exists
  if (!FileUtils::fileExists(fileName))
    throw std::runtime_error("Graph file doesn't exist");

  // get file size
  auto fsize = FileUtils::fileSize(fileName);
  Nd4jLong *ref;
  void *ptrGraph;

  // TODO: check if mmap is supported
  if (true) {
    // mmap this file
    ref = ::mmapFile(nullptr, fileName, fsize);
    ptrGraph = reinterpret_cast<void *>(ref[0]);
  } else {
    // if mmap is not supported - load it directly

    ptrGraph = new uint8_t[fsize];
    auto data = reinterpret_cast<uint8_t *>(ptrGraph);

    FILE *in = fopen(fileName, "rb");
    int cnt = 0;
    int b = 0;
    while (cnt < fsize) {
      b = fread(data + cnt, 1, fsize < 16384 ? fsize : 16384, in);

      cnt += b;
    }
    fclose(in);
  }

  return fromFlatPointer(ptrGraph, memoryManager);
}

Graph Graph::fromFlatPointer(void *ptr,
                             const GraphMemoryManager &memoryManager) {
  // get FlatGraph out of it
  auto fg = GetFlatGraph(reinterpret_cast<uint8_t *>(ptr));

  // return Graph from this FlatGraph
  return Graph(fg, memoryManager);
}

Graph Graph::importFromTensorFlow(const char *fileName) {
  throw std::runtime_error("Graph::importFromTensorFlow() not implemented yet");
  /*
  if (fileName == nullptr)
      return nullptr;

  int fd = open(fileName, O_RDONLY);

  if (fd < 0) {
      nd4j_printf("File not found: [%s]\n", fileName);
      return nullptr;
  }

  nd4j_verbose("Trying to load TF GraphDef from file [%s]\n", fileName);

  tensorflow::GraphDef graphDef;
  bool res = graphDef.ParseFromFileDescriptor(fd);

  // trying to read graph as text
  if(!res) {
      close(fd);
      fd = open(fileName, O_RDONLY);

      google::protobuf::io::FileInputStream fileInput(fd);
      fileInput.SetCloseOnDelete(true);

      if (!google::protobuf::TextFormat::Parse(&fileInput, &graphDef)) {
          nd4j_printf("Failed to read file\n","");
      } else {
          res = true;
      }
  }

  close(fd);

  if (!res)
      return nullptr;

  auto graph = new Graph();
  auto variableSpace = graph->variableSpace();

  std::map<const std::string, int> variablesMap;

  int variablesCounter = 0;
  int nodesCounter = 0;
  nd4j_verbose("Number of nodes in graphDef: %i\n", graphDef.node_size());
  for (int n = 0; n < graphDef.node_size(); n++) {
      auto node = graphDef.node(n);

      // if that's external variable - we put it to variable space
      if (strcmp(TF_VAR, node.op().c_str()) == 0 || strcmp(TF_CONST,
  node.op().c_str()) == 0 || strcmp(TF_INPUT, node.op().c_str()) == 0) {
          nd4j_printf("Variable found: %s\n", node.name().c_str());
          auto variable = new Variable();
          variable->setName(new std::string(node.name().c_str()));
          variable->setId(--variablesCounter);
          variableSpace->putVariable(variable->id(), variable);

          std::pair<const std::string, int> pair(node.name(), variable->id());
          variablesMap.insert(pair);

          // TODO: we might want to have something like that.
          // it basically just gives input validation option, since settles
  expectations for input if (strcmp(TF_INPUT, node.op().c_str()) == 0) continue;

          // checking shape, not applicable to input, since it can vary
          if (node.attr().count("shape")) {
              auto attr = node.attr().at("shape");
              int dims = attr.shape().dim_size();

              if (dims > 0) {
                  std::vector<int> __shape;

                  // we don't have rank1 arrays. vector is 2d.
                  if (dims == 1)
                      __shape.push_back(1);

                  // roll through dimensions
                  for (auto s: attr.shape().dim()) {
                      __shape.push_back((int) s.size()) ;
                  }

                  variable->setNDArray(new NDArray('c', __shape));

                  nd4j_printf("Shape found: %i dims;\n", dims);
                  variable->getNDArray()->printShapeInfo();
              }
          }

          // checking tensor attached
          if (node.attr().count("value")) {
              auto attr = node.attr().at("value");

              // int
              if (attr.tensor().dtype() == ::tensorflow::DataType::DT_INT32) {
                  nd4j_verbose("Int size: %i\n", attr.tensor().int_val_size());

                  Nd4jLong __length = 0;

                  nd4j_verbose("Tensor has shape: %i\n",
  attr.tensor().has_tensor_shape()); if (attr.tensor().has_tensor_shape()) {
                      auto shape = attr.tensor().tensor_shape();
                      int dims = shape.dim_size();

                      if (dims > 0) {
                          std::vector<int> __shape;
                          // we don't have rank1 arrays. vector is 2d.
                          if (dims == 1)
                              __shape.push_back(1);

                          // roll through dimensions
                          for (auto s: shape.dim()) {
                              __shape.push_back((int) s.size());
                          }

                          variable->setNDArray(new NDArray('c', __shape));
                          __length = variable->getNDArray()->lengthOf();

                          nd4j_printf("Tensor shape found: %i dims;\n", dims);
                          variable->getNDArray()->printShapeInfo();
                      }
                  }

                  // it can be valueOf array
                  if (attr.tensor().int_val_size() == 1 && __length > 0) {
                      variable->getNDArray()->assign((T)
  attr.tensor().int_val(0));
                  }
              }
          }
      } else {
          nd4j_verbose("Node id: [%i]; name: [%s]; opName: [%s]\n", n + 1,
  node.name().c_str(), node.op().c_str());

          sd::ops::DeclarableOp *op =
  sd::ops::OpRegistrator::getInstance().getOperationFloat(node.op().c_str());

          if (op == nullptr) {
              nd4j_verbose("Op wasn't found: %s\n", node.op().c_str());
              return nullptr;
          }

          auto jNode = new Node();
          jNode->setName(node.name());
          jNode->setId(++nodesCounter);
          jNode->setCustomOp(op);
          jNode->setBlock(new Block(jNode->id(), variableSpace));

          std::pair<const std::string, int> pair(node.name(), jNode->id());
          variablesMap.insert(pair);

          // multi-output nodes require special treatment
          for (int e = 0; e < op->getOpDescriptor()->getNumberOfOutputs(); e++)
  { std::string deepName(node.name()); deepName += ":" + std::to_string(e); auto
  deepVar = new Variable(); deepVar->setName(&deepName);

              if (e > 0)
                  deepVar->setId(--variablesCounter);
              else
                  deepVar->setId(jNode->id());

              std::pair<const std::string, int> pair(deepName, deepVar->id());
              variablesMap.insert(pair);

              variableSpace->putVariable(deepVar->id(), deepVar);

              std::pair<int, int> nodepair(jNode->id(), e);
              variableSpace->putVariable(nodepair, deepVar);
          }


          printf("             Inputs: [");
          for (int i = 0; i < node.input_size(); i++) {
              nd4j_printf("Trying input: %s\n", node.input(i).c_str());

              // if this fails - we're probably on partial input :)
              if (!variablesMap.count(node.input(i)))
                  return nullptr;

              printf("%s (%i)", node.input(i).c_str(),
  variablesMap.at(node.input(i)));


              jNode->pickInput(variablesMap.at(node.input(i)));
              jNode->getBlock()->pickInput(variablesMap.at(node.input(i)));


              if (i < node.input_size() + 1)
                  printf(", ");
          }
          printf("]\n");

          graph->addNode(jNode);
      }
  }

  return graph;
   */
}

void Graph::addPlaceholder(const std::string &nodeName, DataType dataType,
                           const std::vector<Nd4jLong> &shape) {
  int id = _maxId++;

  _symbolicLookupTable[nodeName] = id;

  auto var = std::make_shared<Variable>(true, dataType, shape);
  var->setName(nodeName);
  _variableSpace.putVariable(id, var);

  _placeholders.emplace_back(nodeName);
}

std::map<std::string, NDArray> Graph::execute(
    const std::map<std::string, NDArray> &dictionary,
    const std::vector<std::string> &outputs,
    const GraphExecutor &executor) const {
  // creating our proxy, we'll use it for actual execution
  VariableProxy proxy(&_variableSpace);

  // first of all we check existence of placeholders in dictionary
  int placeholdersCount = 0;
  for (const auto &v : dictionary) {
    if (_symbolicLookupTable.count(v.first) == 0)
      throw unresolved_input_exception::build("Dictionary entry doesn't exist",
                                              v.first);

    // we also check if arrays provided here do match placeholder restrictions
    // of shape and dtype
    auto var = _variableSpace.getVariable(v.first);
    if (var->dataType() != DataType::ANY &&
        var->dataType() != v.second.dataType())
      throw datatype_exception::build("Placeholder requires another data type",
                                      var->dataType(), v.second.dataType());

    auto shape = v.second.getShapeAsVector();
    if (shape != var->shape())
      throw shape_mismatch_exception::build(
          "Placeholder requires specific shape", var->shape(), shape);

    // update the placeholder
    proxy.putVariable(v.first, var->id(), var->index(), v.second);

    // we must also check if all placeholders were resolved
    placeholdersCount++;
  }

  // TODO: it would be nice if we'll print out unresolved placeholders
  if (placeholdersCount != _placeholders.size())
    throw std::runtime_error("Some placeholders were not resolved");

  // we also must check existence of requested outputs
  for (const auto &v : outputs) {
    if (_symbolicLookupTable.count(v) == 0)
      throw unresolved_output_exception::build("Requested output doesn't exist",
                                               v);
  }

  // execute optimized version of this graph
  auto status = executor.execute(optimizedGraph(), proxy);
  if (status != Status::OK())
    throw graph_execution_exception("Graph execution failed, error code: ", status);

  // fetch outputs from our VariableProxy
  std::map<std::string, NDArray> result;
  for (const auto &v : outputs) {
    if (!proxy.hasVariable(v))
      throw unresolved_output_exception::build(
          "Requested output doesn't exist after execution", v);

    auto var = proxy.getVariable(v);

    // TODO: we want to make sure ManagedDataBuffer doesn't leak here
    result[v] = *var->getNDArray();
  }

  return result;
}

Graph::Graph(const Graph &other) : _memoryManager(other._memoryManager) {
  _configuration = other._configuration;
  _variableSpace = other._variableSpace;
  _stash = other._stash;
  _unmapped = other._unmapped;
  _symbolicLookupTable = other._symbolicLookupTable;
  _built = false;
  _maxId = other._maxId;
}

Graph &Graph::operator=(const Graph &other) noexcept {
  if (this == &other) return *this;

  _configuration = other._configuration;
  _variableSpace = other._variableSpace;
  _stash = other._stash;
  _unmapped = other._unmapped;
  _symbolicLookupTable = other._symbolicLookupTable;
  _built = false;
  _maxId = other._maxId;

  return *this;
}

Graph::Graph(Graph &&other) : _memoryManager(other._memoryManager) {
  _configuration = other._configuration;
  _variableSpace = other._variableSpace;
  _stash = other._stash;

  _unmapped = std::move(other._unmapped);
  _symbolicLookupTable = std::move(other._symbolicLookupTable);

  _built = false;
  _maxId = other._maxId;
}

Graph &Graph::operator=(Graph &&other) noexcept {
  if (this == &other) return *this;

  _configuration = other._configuration;
  _variableSpace = other._variableSpace;
  _stash = other._stash;

  _unmapped = std::move(other._unmapped);
  _symbolicLookupTable = std::move(other._symbolicLookupTable);

  _built = false;
  _maxId = other._maxId;

  return *this;
}

const GraphMemoryManager &Graph::memoryManager() const { return _memoryManager; }

const OptimizedGraph &Graph::optimizedGraph() const {
  std::lock_guard<std::mutex> lock(_optimizedLock);

  // optionally rebuild optimized graph, if it's out of date
  if (_optimized.size() != size())
    _optimized = OptimizedGraph(unmappedNodes(), variableSpace());

  return _optimized;
}
}  // namespace graph
}  // namespace sd
