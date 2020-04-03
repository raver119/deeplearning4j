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
#ifndef SD_OPTIMIZEDGRAPH_H
#define SD_OPTIMIZEDGRAPH_H

#include <graph/execution/OpSequence.h>
#include <graph/execution/ExecutionLayer.h>
#include <memory/GraphMemoryManager.h>
#include <vector>
#include <map>
#include <mutex>

namespace sd {
    namespace graph {
        class Graph;
        class NodeInfo;
        /**
         * This class acts as a topologically sorted & optimized Graph representation, ready for execution
         */
        class SD_EXPORT OptimizedGraph {
        protected:
            // here we store independent OpSequences
            // Graph starts from layer 0, and goes deeper step by step
            // on each layer we can have 1+ OpSequences that can be executed independent
            std::map<uint64_t, ExecutionLayer> _onion;

            GraphMemoryManager *_memoryManager = nullptr;
            Graph *_originalGraph = nullptr;

            std::mutex _mutex;
        public:
            OptimizedGraph(Graph *original);
            ~OptimizedGraph() = default;

            OptimizedGraph(const OptimizedGraph& other) noexcept;

            OptimizedGraph& operator=(const OptimizedGraph& other) noexcept;

            // move constructor
            OptimizedGraph(OptimizedGraph&& other) noexcept;

            // move assignment operator
            OptimizedGraph& operator=(OptimizedGraph&& other) noexcept;


            /**
             * This method returns number of layers within OptimizedGraph
             * @return
             */
            uint64_t layers() const;

            /**
             * This method returns OpSequences stored in a given layer
             * @param index
             * @return
             */
            const ExecutionLayer& layer(uint64_t index) const;

            /**
             * This method allows to append layer to this OptimizedGraph instance
             */
             // FIXME: this method should be removed or made private
            void append(const std::vector<OpSequence> &layer);
            void append(const ExecutionLayer &layer);
            void append(OpSequence &sequence);

            /**
             * This method returns GraphMemoryManager instance that manages this Graph
             * @return
             */
            const GraphMemoryManager& memoryManager() const;

            /**
             * This method returns pointer to original Graph
             * @return
             */
            const Graph& originalGraph() const;

        protected:
            /*
            * optimize original graph
            */
            void optimizedGraph();
            /*
            * Topological graph analysis
            * @param const start node for search
            * @param const reference for nodes infor container
            * @param operation gather
            * @return stop iterating
            */
            bool      topolSearch(const int startNode, const std::unordered_map<int, NodeInfo>& nodesConnections, std::vector<std::vector<OpSequence>>& opSeq) const;
            /*
            * Optimized graph analysis prototyping, gather nodes infor
            * @param reference to node information collector
            * @param reference to start nodes
            * @param reference to input branching nodes (input branching node - atleast 2 internal inputs)
            * @return stop iterating
            */
            bool      opGraphProto(std::unordered_map<int, NodeInfo>& collector, std::set<int>& startNodes, std::set<int>& inBranchingNodes) const;
            /*
            * Define layers and sequence positions based on nodes infor
            * @param reference to node information collector
            * @param node ID
            * @param layer ID
            * @param sequence ID
            * @return stop iterating
            */
            bool      layersSeqDefine(std::unordered_map<int, NodeInfo>& collection, int ID, int layer, int nStartSeq) const;
            /*
            * Initialize container with operations and context
            * @param code reference to node information collector
            * @param reference to opSequence collector
            * @return stop iterating
            */
            bool      initOpSeqContainer(const std::unordered_map<int, NodeInfo>& collection, std::vector<std::vector< OpSequence >>& vOpSeq) const;

        };

        class NodeInfo {
        private:
            std::set<int> sConnections;
            bool bStart;
            bool bInBranching;
            bool bOutBranching;
            int  nLayer;
            int  nSequence;
        public:

            void setStart(bool bValue) { bStart = bValue; }
            void setInBranching(bool bValue) { bInBranching = bValue; }
            void setOutBranching(bool bValue) { bOutBranching = bValue; }

            void reset() { sConnections.clear(); bStart = bInBranching = bOutBranching = false; nLayer = 0; }

            int getLayer() const { return nLayer; }
            void setLayer(int layer) { nLayer = layer; }

            int getSequence() const { return nSequence; }
            void setSequence(int sequence) { nSequence = sequence; }

            void addConnection(int id) { sConnections.emplace(id); }
            const std::set<int>& connections() const { return sConnections; }

            bool isStart() const { return bStart; }
            bool isInBranching() const { return bInBranching; }
            bool isOutBranching() const { return bOutBranching; }

        };

    }
}


#endif //SD_OPTIMIZEDGRAPH_H
