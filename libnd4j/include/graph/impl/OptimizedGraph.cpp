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
// @author Yurii Shyrma (iuriish@yahoo.com)
// @author raver119@gmail.com
//

#include <graph/Graph.h>
#include <graph/OptimizedGraph.h>
#include <forward_list>

namespace sd    {
namespace graph {

///////////////////////////////////////////////////////////////////
// move constructor
OptimizedGraph::OptimizedGraph(OptimizedGraph &&other) noexcept: _sortedGraph(std::move(other._sortedGraph)), _nodesMap(std::move(other._nodesMap)) { }

///////////////////////////////////////////////////////////////////
// move assignment operator
OptimizedGraph& OptimizedGraph::operator=(OptimizedGraph &&other) noexcept {

    if (this == &other)
        return *this;

    _sortedGraph = std::move(other._sortedGraph);
    _nodesMap = std::move(other._nodesMap);

    return *this;
}

///////////////////////////////////////////////////////////////////
OptimizedGraph::OptimizedGraph(const MAP_IMPL<int, Node>& inMap, const VariableSpace& varSpace): _nodesMap(inMap) {

    // for (const auto& p : _nodesMap) {
    //     printf("%s %i, inputs: ", p.second.name().c_str(), p.first);
    //     const auto& inputs = p.second.inputs();
    //     for (int i = 0; i < inputs.size(); ++i)
    //         printf("%i, ", inputs[i].first);
    //     printf("\n");
    // }
    // return;

    struct NodeInfo {
        uint _layerNum = 0;
        std::vector<int> _opSeq = {};
        std::vector<int> _in    = {};
        std::vector<int> _out   = {};
    };

    MAP_IMPL<int, NodeInfo> workMap;  // key is node id, value is class NodeInfo containing auxiliary information (layer number this node belongs to, input/output nodes, OpSequence that starts from this node)

    // create workMap, fill vectors containing input and output nodes per each node, and find start nodes
    std::vector<int> startNodes;
    for (auto& p : _nodesMap) {

        const auto& inputs = p.second.inputs();

        if(p.second.name().find("Exit") != std::string::npos) {
            const int idOfEnter = _nodesMap[_nodesMap[inputs[0].first].inputs()[0].first].inputs()[0].first;
            p.second.setFrameId(_nodesMap[idOfEnter].frameId());
            _nodesMap[idOfEnter].setExitId(p.first);
        }

        for (int i = 0; i < inputs.size(); ++i) {

            if (_nodesMap.count(inputs[i].first) != 0) {              // is op
                _nodesMap[inputs[i].first].pickOutput(p.first, inputs[i].second);
                if(_nodesMap[inputs[i].first].name().find("NextIteration") == std::string::npos) {
                    workMap[inputs[i].first]._out.push_back(p.first);
                    workMap[p.first]._in.push_back(inputs[i].first);
                }
            }
            else {                                              // is variable

                const auto depends = varSpace.getVariable(inputs[i].first).get()->dependencies();

                for (int j = 0; j < depends.size(); ++j) {
                    if(std::find(workMap[p.first]._in.begin(), workMap[p.first]._in.end(), depends[j].first) == workMap[p.first]._in.end()) {
                        _nodesMap[depends[j].first].pickOutput(p.first, depends[j].second);
                        if(_nodesMap[depends[j].first].name().find("NextIteration") == std::string::npos) {
                            workMap[depends[j].first]._out.push_back(p.first);
                            workMap[p.first]._in.push_back(depends[j].first);
                        }
                    }
                }
            }
        }

        if(workMap[p.first]._in.empty())
            startNodes.push_back(p.first);
    }
// printf("\n\n\n\n\n");
    // for (const auto& p : workMap) {
    //     printf("id %i,  inputs: ", p.first);
    //     for (const auto& i : p.second._in)
    //         printf("%i, ", i);
    //     printf("outputs: ");
    //     for (const auto& i : p.second._out)
    //         printf("%i, ", i);
    //     printf("\n");
    // }

    // for (const auto& p : _nodesMap) {
    //     printf("id %i,   inputs ", p.first);
    //     for(const auto& i : p.second.inputs())
    //         printf("%i, ", i.first);
    //     printf("    outputs: ");
    //     for(const auto& i : p.second.outputs())
    //         printf("%i, ", i.first);
    //     printf("\n");
    // }
    // printf("\n\n\n\n\n");


    // collect OpSequences (fill _opSeq)
    std::vector<int> nodesToDelete;
    for (auto& p : workMap) {

        auto& out = p.second._out;
        while(out.size() == 1 && workMap[out[0]]._in.size() == 1) {
            nodesToDelete.push_back(out[0]);
            p.second._opSeq.push_back(out[0]);
            out = std::move(workMap[out[0]]._out);
        }
    }


    // delete nodes present in _opSeq, their ids are already stored in nodesToDelete
    for (const auto& i : nodesToDelete)
        workMap.erase(i);

    // lambda for topological sort
    std::function<void(int,uint,uint&)> visit = [&visit, &workMap] (const int id, const uint layerNum, uint& numOfLayers) {

        if(layerNum <= workMap[id]._layerNum) { return; }
        workMap[id]._layerNum = layerNum;
        if(numOfLayers < layerNum) { numOfLayers = layerNum; }
        for (const auto& nextId : workMap[id]._out)
            visit(nextId, layerNum+1, numOfLayers);
    };

    // perform topological sort
    uint numOfLayers = 0;
    for (const auto& id : startNodes)
        for (const auto& nextId : workMap[id]._out)
            visit(nextId, 1, numOfLayers);

    // gather all nodes belonging to loop and store them in one OpSequence
    const char delimiter = '/';
    for (auto p0 = workMap.begin(); p0 != std::prev(workMap.end()); ++p0) {             // std::prev(workMap.end()) == workMap.end() - 1
        if(!p0->second._opSeq.empty() && p0->second._opSeq[0] == -1)
            continue;
        // const auto& name = _nodesMap[p0->first].name();
        // if(name.find("Enter") == std::string::npos)
        //     continue;
        bool isInLoop = false;
        auto* name = &_nodesMap[p0->first].name();

        if(name->find("Enter") == std::string::npos) {
            for (const auto& id : p0->second._opSeq) {
                name = &_nodesMap[id].name();
                if(name->find("Enter") != std::string::npos) {
                    isInLoop = true;
                    break;
                }
            }
        }
        else
            isInLoop = true;

        if(!isInLoop)
            continue;

        std::string loopName = name->substr(0, name->find(delimiter));    // evaluate name of loop

        for (auto p1 = std::next(p0); p1 != workMap.end(); ++p1) {      // std::next(p0) = p0 + 1

            if(!p1->second._opSeq.empty() && p1->second._opSeq[0] == -1)
                continue;

            isInLoop = false;
            name = &_nodesMap[p1->first].name();

            if(name->find(loopName) == std::string::npos) {
                for (const auto& id : p1->second._opSeq) {
                   name = &_nodesMap[id].name();
                    if(name->find(loopName) != std::string::npos) {
                        isInLoop = true;
                        break;
                    }
                }
            }
            else
                isInLoop = true;

            if(!isInLoop)
                continue;

            p0->second._opSeq.push_back(p1->first);
            p0->second._opSeq.insert(p0->second._opSeq.end(), p1->second._opSeq.begin(), p1->second._opSeq.end());
            p1->second._opSeq.clear();
            p1->second._opSeq.push_back(-1);        // mark node to be neglected
            // p1->second._layerNum = p0->second._layerNum;
        }
    }

    // fill vectors with layers
    std::vector<ExecutionLayer> sortedGraphTemp;//(numOfLayers+1);
    for (const auto& p : workMap) {

        if(!p.second._opSeq.empty() && p.second._opSeq[0] == -1)
            continue;

        OpSequence seq;
        seq.append(_nodesMap.at(p.first), _nodesMap.at(p.first).contextPrototype());

        for (const auto& id : p.second._opSeq)
            seq.append(_nodesMap.at(id), _nodesMap.at(id).contextPrototype());

        while(sortedGraphTemp.size() <= p.second._layerNum)
            sortedGraphTemp.emplace_back(ExecutionLayer());

        sortedGraphTemp[p.second._layerNum].append(std::move(seq));

    }

    // delete empty layers
    for (auto it = sortedGraphTemp.begin(); it != sortedGraphTemp.end(); ++it)
        if(it->width() == 0)
            sortedGraphTemp.erase(it--);

    // check whether there are layers with one OpSequence which in turn contains only one op
    bool isLayerWithOneOp = false;
    for(auto& layer : sortedGraphTemp) {
        if(layer.width() == 1 && layer.at(0).length() == 1) {
            isLayerWithOneOp = true;
            break;
        }
    }

    // fill _sortedGraph
    if(!isLayerWithOneOp) {
        _sortedGraph = std::move(sortedGraphTemp);
    }
    else {
        for (uint i = 0; i < sortedGraphTemp.size();) {

            OpSequence seq;
            while(i < sortedGraphTemp.size() && sortedGraphTemp[i].width() == 1 && sortedGraphTemp[i].at(0).length() == 1)
                seq.append(std::move(sortedGraphTemp[i++].at(0).at(0)));

            if(seq.length() != 0)
                _sortedGraph.emplace_back(ExecutionLayer({seq}));
            else
                _sortedGraph.emplace_back(std::move(sortedGraphTemp[i++]));
        }

    }

    // sort _sortedGraph
    // for (auto& l : _sortedGraph)
    //     l.sortOpSequences();
}

void OptimizedGraph::append(const OpSequence &sequence) {
  _sortedGraph.emplace_back(ExecutionLayer({sequence}));
}

///////////////////////////////////////////////////////////////////
size_t OptimizedGraph::size() const {
  // std::lock_guard<std::mutex> lock(_mutex);

  size_t size = 0;

  std::for_each(_sortedGraph.begin(), _sortedGraph.end(), [&size] (const ExecutionLayer &l) {
      for (int e = 0; e < l.width(); e++) {
        size += l.at(0).length();
      }
  ;});

  return size;
}

void OptimizedGraph::printOut() const {
  for (uint i = 0; i < _sortedGraph.size(); ++i) {
    printf("Layer [%u] {\n", i);
    for (uint j = 0; j < _sortedGraph[i].width(); ++j) {
      printf("   Sequence [%u] {\n", j);
      _sortedGraph[i][j].printOut();
      printf("  }\n");
    }
    printf("}\n");
  }


  printf("And simple print:\n");
  for (int i = 0; i < _sortedGraph.size(); ++i) {
    printf("layer %i: ", i);
    for (int j = 0; j < _sortedGraph[i].width(); ++j) {
      printf("(");
      for (int k = 0; k < _sortedGraph[i][j].length(); ++k) {
        printf("%i, ", _sortedGraph[i][j][k].protoContext().nodeId());
      }
      printf("), ");
    }
    printf("\n");
  }

}


// std::for_each(inMap.begin(), inMap.end(), [] (const std::pair<int, Node> &p) {printf("node id %i \n", p.first);});
    // _sortedGraph = std::vector<ExecutionLayer>(numOfLayers+1);
    // std::vector<std::vector<int>> printGraph = std::vector<std::vector<int>>(numOfLayers+1);
    // for (const auto& p : workMap) {

    //     OpSequence seq;
    //     seq.append(inMap.at(p.first).customOp(), inMap.at(p.first).contextPrototype());
    //     printGraph[p.second._layerNum].push_back(p.first);

    //     for (const auto& id : p.second._opSeq) {
    //         seq.append(inMap.at(id).customOp(), inMap.at(id).contextPrototype());
    //         printGraph[p.second._layerNum].push_back(id);
    //     }

    //     _sortedGraph[p.second._layerNum].append(std::move(seq));
    // }

    // for (int i = 0; i < printGraph.size(); ++i) {
    //     printf("layer %i: ", i);
    //     for (int j = 0; j < printGraph[i].size(); ++j)
    //         printf("%i, ", printGraph[i][j]);
    //     printf("\n");
    // }

    // for (const auto& p : workMap) {
    //     printf("node %i: , layerNum %i: , opSeq: ", p.first,p.second._layerNum);
    //     std::for_each(p.second._opSeq.begin(), p.second._opSeq.end(), [] (const int &j) {printf("%i, ", j);});
    //     printf(",  ins: ");
    //     std::for_each(p.second._in.begin(), p.second._in.end(), [] (const int &j) {printf("%i, ", j);});
    //     printf(",  outs: ");
    //     std::for_each(p.second._out.begin(), p.second._out.end(), [] (const int &j) {printf("%i, ", j);});
    //     printf("\n");
    // }
    // printf("\n-----------------\n");;



// OptimizedGraph::OptimizedGraph(Graph* original) {
//   _originalGraph = original;
//   _memoryManager = const_cast<GraphMemoryManager*>(&original->memoryManager());
//   // create optimized graph
//   createOptimizedGraph();
// }

// OptimizedGraph::OptimizedGraph(const OptimizedGraph& other) noexcept {
//   _onion = other._onion;
//   _memoryManager = other._memoryManager;
//   _originalGraph = other._originalGraph;
// }

// OptimizedGraph& OptimizedGraph::operator=(
//     const OptimizedGraph& other) noexcept {
//   if (this == &other) return *this;

//   _onion = other._onion;
//   _memoryManager = other._memoryManager;
//   _originalGraph = other._originalGraph;

//   return *this;
// }

// OptimizedGraph::OptimizedGraph(OptimizedGraph&& other) noexcept {
//   _onion = std::move(other._onion);
//   _memoryManager = other._memoryManager;
//   _originalGraph = other._originalGraph;
// }

// OptimizedGraph& OptimizedGraph::operator=(OptimizedGraph&& other) noexcept {
//   if (this == &other) return *this;

//   _onion = std::move(other._onion);
//   _memoryManager = other._memoryManager;
//   _originalGraph = other._originalGraph;

//   return *this;
// }





// void OptimizedGraph::append(const std::vector<OpSequence>& layer) {
//   std::lock_guard<std::mutex> lock(_mutex);
//   _onion[_onion.size()] = layer;
//   _size = 0;
// }

// void OptimizedGraph::append(OpSequence& sequence) {
//   append(ExecutionLayer({sequence}));
// }

// void OptimizedGraph::append(const ExecutionLayer& layer) {
//   std::lock_guard<std::mutex> lock(_mutex);
//   _onion[_onion.size()] = layer;
//   _size = 0;
// }

// const GraphMemoryManager& OptimizedGraph::memoryManager() const {
//   return *_memoryManager;
// }

// const Graph& OptimizedGraph::originalGraph() const { return *_originalGraph; }

// bool OptimizedGraph::opGraphProto(std::unordered_map<int, NodeInfo>& collector,
//                                   std::set<int>& startNodes,
//                                   std::set<int>& inBranchingNodes) const {
//   // double check to avoid unstable behavior
//   if (originalGraph().unmappedNodes().empty()) return false;

//   const auto& unmappedNodes = originalGraph().unmappedNodes();
//   // iterate via original graph nodes to gather node information
//   for (const auto& it : unmappedNodes) {
//     const auto& ID = it.first;
//     const auto& inputs = it.second.input();
//     // if node info is not in collecter add it
//     if (collector.find(ID) == collector.end()) collector[ID] = NodeInfo();

//     NodeInfo& parentNode = collector[ID];
//     // count external and internal inputs to find out the type of the node
//     // (start, in-branching, out-branching)
//     int inExCounts = 0, inInternalCounts = 0;
//     for (const auto& in : inputs) {
//       // find input id in original graph
//       if (unmappedNodes.find(in.first) == unmappedNodes.end()) {
//         // count external inputs, all inputs which id is not in unmapped
//         // container will be treaded as external
//         inExCounts++;
//       } else {
//         // count iternal inputs, all inputs that are not in external variable
//         // space will be treated as outputs from other nodes
//         inInternalCounts++;
//         // if node info is not in collector add it
//         if (collector.find(in.first) == collector.end())
//           collector[in.first] = NodeInfo();
//         // input node connection with discovered
//         collector[in.first].addConnection(ID);
//       }
//     }
//     // set operation type
//     parentNode.setType(it.second.opType());

//     // if move then 1 internal input this is in-branching node
//     parentNode.setInBranching(inInternalCounts > 1);
//     // gather start and in-branching nodes for the loop when operations are put
//     // to OpSequence (topolSearch)
//     if (inExCounts == inputs.size()) {
//       startNodes.emplace(ID);
//     } else {
//       if (parentNode.isInBranching()) inBranchingNodes.emplace(ID);
//     }
//   }
//   return true;
// }

// bool OptimizedGraph::topolSearch(
//     const int startNode, std::unordered_map<int, NodeInfo>& collector,
//     std::vector<std::vector<OpSequence>>& opSeq) const {
//   // double check to avoid unstable behavior
//   if (originalGraph().unmappedNodes().empty() || collector.empty())
//     return false;

//   // skip nodes which are not pre-collected and pre-processed
//   auto itParent = collector.find(startNode);
//   if (itParent != collector.end()) {
//     // iterate via start (in-branching) nodes connections in depth
//     for (const auto& itNodes : itParent->second.connections()) {
//       auto itChild = collector.find(itNodes);
//       // double check
//       if (itChild != collector.end()) {
//         // if the child is in-branching node it will be treated as start node or
//         // it was proceed
//         if (itChild->second.isInBranching() || itChild->second.isProcessed()) {
//           continue;
//         }
//         // put operation to OpSequence container
//         const auto it = originalGraph().unmappedNodes().find(itNodes);
//         auto& child = itChild->second;
//         // the layer and sequence are pre-defined in layersSeqDefine method
//         opSeq[child.layer()][child.sequence()].append(
//             it->second.customOp(), it->second.contextPrototype());
//         child.setProcessed();
//         // go to the child node connections
//         topolSearch(itNodes, collector, opSeq);
//       }
//     }
//   }
//   return true;
// }

// void OptimizedGraph::createOptimizedGraph() {
//   // container to store node infor
//   std::unordered_map<int, NodeInfo> collector;
//   // containers to store start and in-branching nodes
//   std::set<int> startNodes, inBranching;
//   // container to store max sequences per layer
//   std::unordered_map<int, int> layersMaxSeq;

//   // optimizing graph prototyping
//   // select start nodes
//   // create connections between nodes
//   // select in-branching nodes ( more then one iternal input -> outputs from
//   // other nodes)
//   if (!opGraphProto(collector, startNodes, inBranching))
//     throw std::runtime_error(
//         "OptimizedGraph::optimizedGraph() - not prototyped!");

//   // next step set the node layer and it sequence in layer
//   // define max layers and max sequence per layer
//   int startSeq = 0;
//   bool bOnlyStartNodes = collector.empty();
//   for (const auto& id : startNodes) {
//     layersMaxSeq[0] = startSeq;
//     // if only start nodes exists they have to be add to connections
//     if (bOnlyStartNodes) {
//       auto node = NodeInfo();
//       node.setLayer(0);
//       node.setProcessed(true);
//       node.setSequence(startSeq);
//       collector[id] = node;
//     } else {
//       layersSeqDefine(collector, id, 0, startSeq, layersMaxSeq);
//     }
//     startSeq++;
//   }

//   // init container to collect operations per node position (layer:sequence)
//   std::vector<std::vector<OpSequence>> vOpSeq;
//   if (!initOpSeqContainer(layersMaxSeq, vOpSeq))
//     throw std::runtime_error(
//         "OptimizedGraph::initOpSeqContainer() - cannot initialize OpSequence, "
//         "not all nodes properly prototyped!");

//   // combine start nodes and in-branching nodes
//   startNodes.insert(inBranching.begin(), inBranching.end());
//   // re-init proceed NodeInfo member to avoid append sequence several times
//   for (auto& it : collector) {
//     it.second.setProcessed(false);
//   }

//   // iterate via start and in-branching nodes
//   for (const auto& id : startNodes) {
//     const auto it = originalGraph().unmappedNodes().find(id);
//     auto& nodeInfo = collector[id];
//     // append start/in-branching node operation to sequence
//     if (!nodeInfo.isProcessed()) {
//       vOpSeq[nodeInfo.layer()][nodeInfo.sequence()].append(
//           it->second.customOp(), it->second.contextPrototype());
//       nodeInfo.setProcessed();
//     }

//     // search in depth via connections of "start" node
//     if (!topolSearch(id, collector, vOpSeq))
//       throw std::runtime_error(
//           "OptimizedGraph::topolSearch() - cannot run topological search, "
//           "inputs incorrect!");
//   }
//   // put results to optimized graph
//   for (auto& vSeq : vOpSeq) {
//     this->append(vSeq);
//   }
// }

// bool OptimizedGraph::initOpSeqContainer(
//     const std::unordered_map<int, int>& layersMaxSeq,
//     std::vector<std::vector<OpSequence>>& vOpSeq) const {
//   // double check to avoid unstable behavior
//   if (layersMaxSeq.empty()) return false;
//   // pre-init op-sequence size layers/per-layer sequence
//   vOpSeq.resize(layersMaxSeq.size());
//   for (const auto& it : layersMaxSeq) {
//     vOpSeq[it.first].resize(it.second + 1);
//   }
//   return true;
// }

// bool OptimizedGraph::layersSeqDefine(
//     std::unordered_map<int, NodeInfo>& collection, int ID, int layer,
//     int startSeq, std::unordered_map<int, int>& layersMaxSeq) const {
//   // double check to avoid unstable behavior
//   auto parent = collection.find(ID);
//   if (parent == collection.end()) return false;

//   // if node was proceed and the current layer is less of it own return
//   if (parent->second.isProcessed() && parent->second.layer() >= layer)
//     return true;

//   // put layer and sequence to container that collects layers and max sequence
//   // per layer
//   auto layerFound = layersMaxSeq.find(layer);
//   if (layerFound == layersMaxSeq.end()) {
//     // if layer was not treated before, create pair for it
//     layersMaxSeq[layer] = 0;
//     // set sequence value to 0, as this is first sequence in layer
//     startSeq = 0;
//   } else {
//     // if node sequence position was not checked use it for max sequence
//     // selection sequence have to be incremented as max + 1, without any jumps
//     if (startSeq > (layerFound->second + 1)) startSeq = layerFound->second + 1;

//     layerFound->second =
//         (layerFound->second < startSeq && parent->second.sequence() < 0)
//             ? startSeq
//             : layerFound->second;
//   }

//   // double check if the layer is higher and set node layer
//   if (parent->second.layer() < layer) parent->second.setLayer(layer);
//   // double check if sequence was init, if not set current sequence
//   if (parent->second.sequence() < 0) parent->second.setSequence(startSeq);
//   // set is node out-branching
//   parent->second.setOutBranching(parent->second.connections().size() > 1);
//   // set that node was processed, to avoid it double processing (only for some
//   // cases it can be processed several times)
//   parent->second.setProcessed();

//   // if current node is out-branching it childs will be put to next layer
//   if (parent->second.isOutBranching() && !parent->second.isLogic()) layer++;

//   // childs sequence position have to start from max defined sequence position
//   // in layer or if it is first node in layer from 0
//   int seq = (layersMaxSeq.find(layer) == layersMaxSeq.end())
//                 ? 0
//                 : layersMaxSeq[layer];
//   // if parent is out-branching node sequence have to be increment
//   // on the next stage the sequence value will be double checked with max per
//   // layer todo check logic part maybe here have to be check operation class
//   // (something likke Switch, If, While etc) probably for each of them could be
//   // other behavior
//   seq = (parent->second.isOutBranching() && !parent->second.isLogic()) ? seq + 1
//                                                                        : seq;

//   // loop via childs (connected nodes)
//   for (const auto& id : parent->second.connections()) {
//     // double check to avoid unstable behavior
//     auto child = collection.find(id);
//     if (child == collection.end()) return false;

//     // in case parent was not out-branching node but child is in branching it
//     // will be put to next layer todo check logic part
//     if (!parent->second.isOutBranching() && child->second.isInBranching() &&
//         !child->second.isLogic())
//       layer++;

//     // move in depth of connections
//     layersSeqDefine(collection, id, layer, seq, layersMaxSeq);
//     // increment sequence as childs are on the one layer in case if child was
//     // not processed earlier todo check logic part
//     if (!parent->second.isLogic()) seq++;
//   }

//   return true;
// }



}  // namespace graph
}  // namespace sd
