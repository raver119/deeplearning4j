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

#ifndef LIBND4J_GNODE_H
#define LIBND4J_GNODE_H

#include <atomic>
#include <system/pointercast.h>
#include <string>
#include <array/NDArray.h>
#include "Context.h"
#include <ops/declarable/DeclarableOp.h>
#include <graph/generated/node_generated.h>


namespace sd {
    namespace graph {


        class Graph;

        class SD_EXPORT Node {
        protected:
            // TODO: this field must be removed
            sd::DataType _dataType;

            OpType _opType;
            ContextPrototype _protoContext;
            Nd4jLong _opNum;
            int _id = 0;
            std::vector<std::pair<int, int>> _input;
            std::vector<std::pair<int, int>> _output;
            std::vector<int> _dimensions;

            std::vector<int> _referencedBy;

            std::string _name;

            // many ops require extra parameters to run
            double *_extraParams = nullptr;

            bool _hasExternalOutputs;
            bool _hasExternalInputs;
            bool _hasInternalOutputs;
            bool _hasInternalInputs;

            // this field is used to check, if op should be used in-place (so it can/will modify its inputs)
            bool _isInplace = false;

            OpClass _opClass;

            // these fields are used to store embedded CustomOps and Graph in case of Graph-in-Graph scenario
            Graph * _graph= nullptr;
            std::shared_ptr<sd::ops::DeclarableOp> _customOp;

            // each node can be active or inactive, if used with divergents, like IF statements
            bool _active = true;

            // meh
            mutable bool _removable = true;

            // these fields contain information about Scope these ops are related to
            int _scope_id = 0;
            std::string _scope_name;

            // TODO: these 3 fields should be removed
            int _rewindNode = -1;
            std::pair<int, int> _rewindLayer = {-1, -1};
            Nd4jLong _frameId = -1;

        public:

            explicit Node(const sd::ops::DeclarableOp &op, const std::string &nodeName = {}, const std::vector<double> &tArgs = {}, const std::vector<Nd4jLong> &iArgs = {}, const std::vector<bool> &bArgs = {}, const std::vector<DataType> &dArgs = {});
            explicit Node(const std::string &opName, const std::string &nodeName = {}, const std::vector<double> &tArgs = {}, const std::vector<Nd4jLong> &iArgs = {}, const std::vector<bool> &bArgs = {}, const std::vector<DataType> &dArgs = {});
            explicit Node(const FlatNode *node);
            ~Node();

            /*
             * FIXME: deprecated methods, to be removed
             */
            explicit Node(const std::string &opName, const std::string &nodeName, const int id, const std::vector<std::string> &inputs = {}, const std::vector<double> &tArgs = {}, const std::vector<Nd4jLong> &iArgs = {});
            explicit Node(const std::string &opName, const int id = 0, const std::vector<std::pair<int,int>> &inputs = {}, const std::vector<double> &tArgs = {}, const std::vector<Nd4jLong> &iArgs = {});
            explicit Node(sd::ops::DeclarableOp *customOp, int id = 0, std::initializer_list<int> input = {}, std::initializer_list<int> output = {},  std::initializer_list<int> dimensions = {}, float scalar = 0.0f, std::initializer_list<double> tArgs = {}, std::initializer_list<int> iArgs = {});
            explicit Node(std::shared_ptr<sd::ops::DeclarableOp> customOp, int id = 0, std::initializer_list<int> input = {}, std::initializer_list<int> output = {},  std::initializer_list<int> dimensions = {}, float scalar = 0.0f, std::initializer_list<double> tArgs = {}, std::initializer_list<int> iArgs = {});
            explicit Node(OpType opType = OpType_TRANSFORM_SAME, int opNum = 0, int id = 0, std::initializer_list<int> input = {}, std::initializer_list<int> output = {},  std::initializer_list<int> dimensions = {}, float scalar = 0.0f, std::initializer_list<double> tArgs = {}, std::initializer_list<int> iArgs = {});


            Node(const Node& other) noexcept;

            Node& operator=(const Node& other) noexcept;

            // move constructor
            Node(Node&& other) noexcept;

            // move assignment operator
            Node& operator=(Node&& other) noexcept;

            bool equals(Node *other) const;

            sd::DataType dataType();
            const ContextPrototype& protoContext() const;
            OpType opType() const;
            Nd4jLong opNum() const;
            int id() const;
            const std::vector<std::pair<int,int>>& input() const;
            const std::vector<std::pair<int, int>>& output() const;

            Nd4jLong getFrameId();
            void setFrameId(Nd4jLong frameId);

            int getRewindNode();
            void setRewindNode(int nodeId);

            std::pair<int, int>& getRewindLayer();
            void setRewindLayer(int layerId, int stepId = 0);

            void setId(int id);

            double *extraParams();

            bool isMultiInput();
            bool isMultiOutput();

            bool isRemovable() const;
            void markRemovable(bool reallyRemovable) const;

            bool isDivergencePoint();
            void setActive(bool reallyActive);
            bool isActive();

            bool hasExternalOutputs();
            bool hasExternalInputs();
            bool hasInternalOutputs();
            bool hasInternalInputs();

            void pickOutputOnce(int outputId);
            void pickOutput(int outputId);
            void pickOutput(int nodeId, int outputId);
            void pickExternalOutput(int outputId);
            void pickInput(int inputId);
            void pickInput(int nodeId, int outputId);
            void pickInput(std::pair<int,int>& id);
            void pickInput(const std::string &id);

            void setName(std::string *name);
            void setName(const std::string& name);
            const std::string& getName() const;
            const std::string& name() const;

            int totalReferences();
            void addReference(int nodeId);

            void setContextPrototype(const ContextPrototype &block);
            const ContextPrototype& contextPrototype() const;
            bool hasBlockAttached();

            void setCustomOp(std::shared_ptr<sd::ops::DeclarableOp> customOp);
            std::shared_ptr<sd::ops::DeclarableOp> customOp() const;
            bool hasCustomOp() const;

            void setGraph(Graph* graph = nullptr);
            Graph* graph() const;
            bool hasGraphEmbedded() const;

            bool isInplace();
            void markInplace(bool reallyInplace);


            OpClass getOpClass();

            // these methods are used for internal profiling
            void setOuterTime(Nd4jLong time);
            void setInnerTime(Nd4jLong time);

            // methods related to scopes
            bool isScoped();
            void setScopeInfo(int id, const char* name = nullptr);
            int scopeId();
            std::string* scopeName();

            void setOpType(OpType opType);

            // clone Node
            Node* clone();

            template <typename T>
            Node* asT();

            FORCEINLINE void pullValues(Node *other) {
                this->_dataType = other->dataType();
                this->_protoContext = other->protoContext();
                this->_hasExternalInputs = other->hasExternalInputs();
                this->_hasExternalOutputs = other->hasExternalOutputs();
                this->_hasInternalInputs = other->hasInternalInputs();
                this->_hasInternalOutputs = other->hasInternalOutputs();

                this->markInplace(other->isInplace());
                this->setActive(other->isActive());
                this->setScopeInfo(other->scopeId(), other->scopeName()->c_str());

                for (auto &v: other->input())
                    this->_input.emplace_back(v);

                for (auto &v: other->output())
                    this->_output.emplace_back(v);
            }

            static std::shared_ptr<sd::ops::DeclarableOp> buildOpByType(OpType opType, int numInputs, int numIArgs, int numTArgs, int opNum);
            static void deleteOpByType(OpType opType, void *op);
        };
    }
}

#endif //LIBND4J_GNODE_H
