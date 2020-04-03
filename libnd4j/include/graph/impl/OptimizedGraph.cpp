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

#include <graph/OptimizedGraph.h>
#include <graph/Graph.h>

namespace sd {
    namespace graph {
        OptimizedGraph::OptimizedGraph(Graph *original) {
            _originalGraph = original;
            _memoryManager = const_cast<GraphMemoryManager*>(&original->memoryManager());
            optimizedGraph();
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


        bool   OptimizedGraph::opGraphProto(std::unordered_map<int, NodeInfo>& collector, std::set<int>& startNodes, std::set<int>& inBranchingNodes) const {

            if (originalGraph().unmappedNodes().empty())
                return false;

            for (const auto& it : originalGraph().unmappedNodes()) {

                const auto& ID = it.first;
                const auto& inputs = it.second.input();

                if (collector.find(ID) == collector.end())
                    collector[ID] = NodeInfo();

                NodeInfo& parentNode = collector[ID];

                int inExCounts = 0, inInternalCounts = 0;
                for (auto in = inputs.begin(); in != inputs.end(); ++in) {
                    if (originalGraph().variableSpace().hasVariable(in->first, 0)) {
                        inExCounts++;
                    }
                    else {
                        inInternalCounts++;
                        if (collector.find(in->first) == collector.end())
                            collector[in->first] = NodeInfo();
                        collector[in->first].addConnection(ID);
                    }
                }

                parentNode.setInBranching((inInternalCounts == inputs.size() && inInternalCounts > 1));

                parentNode.setStart(inExCounts == inputs.size());
                parentNode.setSequence(-1);

                if (parentNode.isStart()) {
                    parentNode.setLayer(0);
                    startNodes.emplace(ID);
                }
                else {
                    if (parentNode.isInBranching())
                        inBranchingNodes.emplace(ID);
                }
            }
            return true;
        }

        bool  OptimizedGraph::topolSearch(const int startNode, const std::unordered_map<int, NodeInfo>& collector,
            std::vector<std::vector<OpSequence> >& opSeq) const {

            if (originalGraph().unmappedNodes().empty())
                return false;

            auto itParent = collector.find(startNode);
            if (itParent != collector.end()) {

                for (const auto& itNodes : itParent->second.connections()) {

                    auto itChild = collector.find(itNodes);

                    if (itChild != collector.end()) {

                        if (itChild->second.isInBranching()) {
                            return true;
                        }

                        const auto it = originalGraph().unmappedNodes().find(itNodes);
                        const auto& child = itChild->second;
                        opSeq[child.getLayer()][child.getSequence()].append(it->second.customOp(), it->second.contextPrototype());

                        topolSearch(itNodes, collector, opSeq);
                    }
                }
            }
            return true;
        }

        void  OptimizedGraph::optimizedGraph() {

            std::unordered_map<int, NodeInfo> collector;
            std::set<int> startNodes, inBranching;

            // todo check this will be empty Optimized graph 
            if (!opGraphProto(collector, startNodes, inBranching))
                throw std::runtime_error("OptimizedGraph::optimizedGraph() - not prototyped");

            int startSeq = 0;
            for (const auto& id : startNodes) {
                layersSeqDefine(collector, id, 0, startSeq);
                startSeq++;
            }

            std::vector<std::vector<OpSequence>> vOpSeq;
            initOpSeqContainer(collector, vOpSeq);

            startNodes.insert(inBranching.begin(), inBranching.end());

            for (const auto& id : startNodes) {

                const auto it = originalGraph().unmappedNodes().find(id);
                const auto& nodeInfo = collector[id];
                vOpSeq[nodeInfo.getLayer()][nodeInfo.getSequence()].append(it->second.customOp(), it->second.contextPrototype());

                topolSearch(id, collector, vOpSeq);
            }

            for (auto& vSeq : vOpSeq) {
                this->append(vSeq);
            }
        }

        bool   OptimizedGraph::initOpSeqContainer(const std::unordered_map<int, NodeInfo>& collection, std::vector<std::vector< OpSequence >>& vOpSeq) const {

            if (collection.empty())
                return false;

            int layer = 0;
            std::vector<int> vSeq;
            for (const auto& node : collection) {

                int nodeLayer = node.second.getLayer();
                int nodeSeq = node.second.getSequence();
                if (layer < nodeLayer)
                    layer = nodeLayer;

                if (vSeq.size() < nodeLayer + 1) {
                    vSeq.resize(nodeLayer + 1, 0);
                }
                // each layer will have it own max sequence
                if (vSeq[nodeLayer] < nodeSeq)
                    vSeq[nodeLayer] = nodeSeq;
            }

            vOpSeq.resize(layer + 1);
            for (int i = 0; i <= layer; ++i) {
                vOpSeq[i].resize(vSeq[i] + 1);
            }
            return true;
        }

        bool OptimizedGraph::layersSeqDefine(std::unordered_map<int, NodeInfo>& collection, int ID, int layer, int startSeq) const {

            auto parent = collection.find(ID);
            if (parent == collection.end())
                return false;

            if (parent->second.isInBranching()) {
                layer++;
                if (startSeq > 0)
                    startSeq--;
            }

            parent->second.setLayer(layer);
            // sequence have to be init once
            if (parent->second.getSequence() < 0)
                parent->second.setSequence(startSeq);

            parent->second.setOutBranching(parent->second.connections().size() > 1);
            if (parent->second.isOutBranching())
                layer++;

            for (const auto& id : parent->second.connections()) {

                auto child = collection.find(id);

                layersSeqDefine(collection, id, layer, startSeq);

                if (parent->second.isOutBranching())
                    startSeq++;
            }

            return true;
        }
    }
}