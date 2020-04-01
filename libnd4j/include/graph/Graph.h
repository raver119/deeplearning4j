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

#ifndef LIBND4J_GRAPH_H
#define LIBND4J_GRAPH_H

#include <list>
#include <algorithm>
#include <unordered_map>
#include <map>
//#include <NDArray.h>
#include <graph/Node.h>
#include <graph/Stash.h>
#include <graph/Scope.h>
#include <graph/Variable.h>
#include <graph/VariableSpace.h>
#include <graph/generated/node_generated.h>
#include <graph/generated/graph_generated.h>
#include <graph/generated/config_generated.h>
#include <graph/ExecutorConfiguration.h>
#include <ops/declarable/OpDescriptor.h>
#include <graph/execution/GraphExecutor.h>
#include <graph/OptimizedGraph.h>
#include <memory/GraphMemoryManager.h>

namespace sd {
    namespace graph {

        class SD_EXPORT Graph {
        protected:
            ExecutorConfiguration _configuration;
            VariableSpace *_variableSpace;
            Stash _stash;

            MAP_IMPL<int, Node> _unmapped;

            // string -> id conversion table
            MAP_IMPL<std::string, int> _symbolicLookupTable;

            std::mutex _mutexPreprocessing;
            std::atomic<bool> _built;

            // we want to know last node id
            int _maxId = 1;

            const GraphMemoryManager &_memoryMaager;

////////////////////////////////////////
            Nd4jStatus validateNode(Node *node);

            int idByName(const std::string &nodeName) const;

            void printOutNode(const Node &node) const;

            std::vector<std::string> _placeholders;
        public:
            Graph(const FlatGraph *flatGraph = nullptr, VariableSpace *variableSpace = nullptr, const GraphMemoryManager &memoryManager = GraphMemoryManager());

            ~Graph();

            Graph(const Graph& other);

            Graph& operator=(const Graph& other) noexcept;

            // move constructor
            Graph(Graph&& other);

            // move assignment operator
            Graph& operator=(Graph&& other) noexcept;

            /**
             * Methods that allow Graph imports
             */
            static Graph importFromTensorFlow(const char *fileName);
            static Graph fromFlatBuffers(const char *fileName, const GraphMemoryManager &memoryManager = GraphMemoryManager());
            static Graph fromFlatPointer(void *ptr, const GraphMemoryManager &memoryManager = GraphMemoryManager());

            // method that'll print out graph
            Nd4jStatus validate();

            // this method returns total number of nodes in this graph
            int size() const;

            int numberOfPlaceholders() const;

            const std::vector<Variable*>& getPlaceholders() const;

            /**
             * This method returns pointer to thread_local VariableSpace
             * @return
             */
            VariableSpace *getVariableSpace() const;

            /**
             * These methods add given node to the graph
             * @param node
             */
            void addNode(Node &&node, const std::initializer_list<std::string> &inputs);

            void addNode(Node &node, const std::initializer_list<std::string> &inputs);
            void addNode(Node &node, const std::initializer_list<int> &inputs);
            void addNode(Node &node, const std::initializer_list<std::pair<int, int>> &inputs);


            void addVariable(const std::string &name, NDArray &array);
            void addVariable(const std::string &name, NDArray &&array);

            /**
             * This method allows to add placeholder with some pre-defined properties
             */
            void addPlaceholder(const std::string &nodeName, const DataType dataType = sd::DataType::ANY, const std::vector<Nd4jLong> &shape = {});


            /**
             * This method returns pointer to ExecutorConfiguration
             *
             * @return
             */
            const ExecutorConfiguration& getExecutorConfiguration() const;

            /**
             * This method prints out Graph op-by-op, and respective inputs
             */
            void printOut();

            /**
             * This method returns clone of the graph
             */
            Graph* clone() const;

            /**
             * This method returns clone of the graph, backed by VariableProxy instead of VariableSpace
             */
            Graph cloneWithProxy() const;

            /**
             * This method removes reference to VariableSpace from this Graph
             */
            void forgetVariableSpace();

            /**
             * This method returns hash of given Graph instance
             */
            Nd4jLong hashCode() const;

            void replaceState(VariableSpace *state, const ExecutorConfiguration &configuration);

            FORCEINLINE bool built();

            OptimizedGraph optimizedGraph() const;

            /**
             * This method executes this Graph instance and returns execution results
             * @param dictionary
             * @return
             */
            std::map<std::string, NDArray> execute(const std::map<std::string, NDArray> &dictionary = {}, const std::vector<std::string> &outputs = {}, const GraphExecutor &executor = GraphExecutor()) const;
protected:
            /*
            * Topological graph analysis
            * @param const start node for search
            * @param const reference to list of nodes without external inputs
            * @param const node positions in _handler
            * @param operation gather
            * @return stop iterating
            */
            bool      topolSearch(const int startNode, const std::set<int>& nodeBranches,
                const std::unordered_map<int, int>& positions, OpSequence& opSeq) const;
        };


        FORCEINLINE bool Graph::built() {
            return _built.load();
        }
    }
}

#endif //LIBND4J_GRAPH_H
