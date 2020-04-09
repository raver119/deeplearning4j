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
// @author oleg.semeniv@gmail.com
//

#include <graph/OptimizedGraph.h>
#include <graph/Graph.h>

namespace sd {
    namespace graph {
        OptimizedGraph::OptimizedGraph(Graph *original) {
            _originalGraph = original;
            _memoryManager = const_cast<GraphMemoryManager*>(&original->memoryManager());
            // create optimized graph
            createOptimizedGraph();
        }

        OptimizedGraph::OptimizedGraph(const OptimizedGraph &other) noexcept {
            _onion = other._onion;
            _memoryManager = other._memoryManager;
        }

        OptimizedGraph &OptimizedGraph::operator=(const OptimizedGraph &other) noexcept {
            if (this == &other)
                return *this;

            _onion = other._onion;
            _memoryManager = other._memoryManager;

            return *this;
        }

        OptimizedGraph::OptimizedGraph(OptimizedGraph &&other) noexcept {
            _onion = std::move(other._onion);
            _memoryManager = other._memoryManager;
        }

        OptimizedGraph &OptimizedGraph::operator=(OptimizedGraph &&other) noexcept {
            if (this == &other)
                return *this;

            _onion = std::move(other._onion);
            _memoryManager = other._memoryManager;

            return *this;
        }

        uint64_t OptimizedGraph::layers() const {
            return _onion.size();
        }

        const ExecutionLayer &OptimizedGraph::layer(uint64_t index) const {
            return _onion.at(index);
        }

        void OptimizedGraph::append(const std::vector<OpSequence> &layer) {
            std::lock_guard<std::mutex> lock(_mutex);
            _onion[_onion.size()] = layer;
        }

        void OptimizedGraph::append(OpSequence &sequence) {
            append(ExecutionLayer({sequence}));
        }

        void OptimizedGraph::append(const ExecutionLayer &layer) {
            std::lock_guard<std::mutex> lock(_mutex);
            _onion[_onion.size()] = layer;
        }

        const GraphMemoryManager &OptimizedGraph::memoryManager() const {
            return *_memoryManager;
        }

        const Graph &OptimizedGraph::originalGraph() const {
            return *_originalGraph;
        }

        bool   OptimizedGraph::opGraphProto(std::unordered_map<int, NodeInfo>& collector, std::set<int>& startNodes, 
                                            std::set<int>& inBranchingNodes) const {
            
            // double check to avoid unstable behavior
            if (originalGraph().unmappedNodes().empty())
                return false;

            // iterate via original graph nodes to gather node information
            for (const auto& it : originalGraph().unmappedNodes()) {

                const auto& ID = it.first;
                const auto& inputs = it.second.input();
                // if node info is not in collecter add it
                 if (collector.find(ID) == collector.end())
                    collector[ID] = NodeInfo();

                NodeInfo& parentNode = collector[ID];

                int inExCounts = 0, inInternalCounts = 0;
                for (const auto& in : inputs) {

                    // todo have to be correctly defined what input is external and what internal
                    if (originalGraph().variableSpace().hasVariable(in.first, 0)) {
                        // count external inputs, all inputs that are in
                        // varable space will be treated as external inputs
                        inExCounts++;
                    }
                    else {
                        // count iternal inputs, all inputs that are not in external variable space 
                        // will be treated as outputs from other nodes
                        inInternalCounts++;
                        // if node info is not in collector add it
                        if (collector.find(in.first) == collector.end())
                            collector[in.first] = NodeInfo();
                        
                        collector[in.first].addConnection(ID);
                    }
                }
                // if move then 1 internal input this is in-branching node
                parentNode.setInBranching( inInternalCounts > 1);
                // gather start and in-branching nodes for the loop when operations are put to OpSequence (topolSearch)
                if (inExCounts == inputs.size()) {
                    startNodes.emplace(ID);
                }
                else {
                    if (parentNode.isInBranching())
                        inBranchingNodes.emplace(ID);
                }
            }
            return true;
        }

        bool  OptimizedGraph::topolSearch(const int startNode, std::unordered_map<int, NodeInfo>& collector,
                                           std::vector<std::vector<OpSequence> >& opSeq) const {
            
            // double check to avoid unstable behavior
            if (originalGraph().unmappedNodes().empty() || collector.empty() )
                return false;
            
            // skip nodes which are not pre-collected and pre-processed
            auto itParent = collector.find(startNode);
            if (itParent != collector.end()) {
                // iterate via start (in-branching) nodes connections in depth
                for (const auto& itNodes : itParent->second.connections()) {

                    auto itChild = collector.find(itNodes);

                    if (itChild != collector.end()) {
                        // if the child is in-branching node it will be treated as start node
                        if (itChild->second.isInBranching() || itChild->second.isProcessed()) { // proceed
                            continue;
                        }
                        // put operation to OpSequence container
                        const auto it = originalGraph().unmappedNodes().find(itNodes);
                        auto& child = itChild->second;
                        // the layer and sequence are pre-defined in layersSeqDefine method
                        opSeq[child.layer()][child.sequence()].append(it->second.customOp(), it->second.contextPrototype());
                        child.setProcessed();
                        // go to the child node connections
                        topolSearch(itNodes, collector, opSeq);
                    }
                }
            }
            return true;
        }

        void  OptimizedGraph::createOptimizedGraph() {
             
            // container to store node infor
            std::unordered_map<int, NodeInfo> collector;
            // containers to store start and in-branching nodes
            std::set<int> startNodes, inBranching;
            // container to store max sequences per layer
            std::unordered_map<int, int> layersMaxSeq;

            // optimizing graph prototyping
            // select start nodes
            // create connections between nodes
            // select in-branching nodes ( more then one iternal input -> outputs from other nodes)
            if (!opGraphProto(collector, startNodes, inBranching))
                throw std::runtime_error("OptimizedGraph::optimizedGraph() - not prototyped!");
            
            // next step set the node layer and it sequence in layer
            // define max layers and max sequence per layer
            int startSeq = 0;
            for (const auto& id : startNodes) {
                layersMaxSeq[0] = startSeq;
                // if only start nodes exists they have to be add to connections
                if (bOnlyStartNodes) {
                    auto node = NodeInfo();
                    node.setLayer(0);
                    node.setProcessed(true);
                    node.setSequence(startSeq);
                    collector[id] = node;
                }
                else {
                    if (!layersSeqDefine(collector, id, 0, startSeq, layersMaxSeq))
                        throw std::runtime_error("OptimizedGraph::layersSeqDefine() - not all nodes properly prototyped!");
                }
                startSeq++;
            }
            
            // init container to collect operations per node position (layer:sequence)
            std::vector<std::vector<OpSequence>> vOpSeq;
            if(!initOpSeqContainer(layersMaxSeq, vOpSeq))
               throw std::runtime_error("OptimizedGraph::initOpSeqContainer() - cannot initialize OpSequence, not all nodes properly prototyped!");

            // combine start nodes and in-branching nodes
            startNodes.insert(inBranching.begin(), inBranching.end());
            // re-init proceed NodeInfo member to avoid append sequence several times
            for(auto& it : collector){
                it.second.setProcessed(false);
            }  

            // iterate via start and in-branching nodes
            for (const auto& id : startNodes) {
                 
                const auto it = originalGraph().unmappedNodes().find(id);
                auto& nodeInfo = collector[id];
                // append only not processed nodes
                if(!nodeInfo.isProcessed()){
                   vOpSeq[nodeInfo.layer()][nodeInfo.sequence()].append(it->second.customOp(), it->second.contextPrototype());
                   nodeInfo.setProcessed();
                }                
                // search in depth via connections of "start" node
                if(!topolSearch(id, collector, vOpSeq))
                    throw std::runtime_error("OptimizedGraph::topolSearch() - cannot run topological search, inputs incorrect!");
            }
            // put results to optimized graph
            for (auto& vSeq : vOpSeq) {
                this->append(vSeq);
            }
        }

        bool   OptimizedGraph::initOpSeqContainer(const std::unordered_map<int, int>& layersMaxSeq, std::vector<std::vector< OpSequence >>& vOpSeq) const {
            
            // double check to avoid unstable behavior
            if (layersMaxSeq.empty())
                return false;

            vOpSeq.resize(layersMaxSeq.size());
            for (const auto& it : layersMaxSeq) {
                vOpSeq[it.first].resize(it.second + 1);
            }
            return true;
        }

        bool OptimizedGraph::layersSeqDefine(std::unordered_map<int, NodeInfo>& collection, int ID, int layer, int startSeq, 
                                             std::unordered_map<int, int>& layersMaxSeq) const {
            
            // double check to avoid unstable behavior
            auto parent = collection.find(ID);
            if (parent == collection.end())
                return false;
            
            // if node was proceed and the current layer is less of it own return
            if(parent->second.isProcessed() && parent->second.layer() >= layer)
                return true;
            
            // put layer and sequence to container that collects layers and max sequence per layer
            auto layerFound = layersMaxSeq.find(layer);
            if(layerFound == layersMaxSeq.end()){
                // if layer was not treated before, create pair for it
                layersMaxSeq[layer] = 0;
                // set sequence value to 0, as this is first sequence in layer
                startSeq = 0;
            }
            else{
                // if node sequence position was not checked use it for max sequence selection
                if(startSeq > (layerFound->second + 1))
                   startSeq = layerFound->second + 1;
                   
                layerFound->second = (layerFound->second < startSeq && parent->second.sequence() < 0) ? startSeq : layerFound->second;
            }

            // double check if the layer is higher and set node layer
            if(parent->second.layer() < layer)
               parent->second.setLayer(layer);
            // double check if sequence was init, if not set current sequence
            if(parent->second.sequence() < 0)
                parent->second.setSequence(startSeq);
            // set is node out-branching 
            parent->second.setOutBranching(parent->second.connections().size() > 1);
            // set that node was processed, to avoid it double processing (only for some cases it can be processed several times)
            parent->second.setProcessed();
            
            // if current node is out-branching it childs will be put to next layer
            if (parent->second.isOutBranching())
                layer++;
            
            // childs sequence position have to start from max defined sequence position in layer
            // or if it is first node in layer from 0
            int seq = (layersMaxSeq.find(layer) == layersMaxSeq.end()) ? 0 : layersMaxSeq[layer];
            // if parent is out-branching node sequence have to be increment 
            // on the next stage the sequence value will be double checked with max per layer
            seq = parent->second.isOutBranching() ? seq + 1 : seq;
            
            // loop via childs (connected nodes)
            for (const auto& id : parent->second.connections()) {
                // double check to avoid unstable behavior
                auto child = collection.find(id);
                if(child == collection.end())
                   return false;
                
                // in case parent was not out-branching node but child is in branching it will be put to next layer
                if (!parent->second.isOutBranching() && child->second.isInBranching())
                     layer++;
                
                // move in depth of connections
                layersSeqDefine(collection, id, layer, seq, layersMaxSeq);
                // increment sequence as childs are on the one layer in case if child was not processed earlier 
                seq++;
            }

            return true;
        }
    }
}
