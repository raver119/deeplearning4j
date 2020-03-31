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

#include <graph/VariableSpace.h>

namespace sd {
    namespace graph {
        class SD_EXPORT VariableProxy: public VariableSpace {
        protected:
            VariableSpace* _backed = nullptr;
            VariableSpace* _current = nullptr;
        public:
            explicit VariableProxy(VariableSpace* reference);
            ~VariableProxy();

            virtual VariableSpace& operator=(const VariableSpace& other);

            virtual int numberOfPlaceholders() override;
            virtual std::vector<Variable*>* getPlaceholders() override;

            virtual sd::memory::Workspace *workspace();

            virtual bool hasExternalVariable(int it) override;
            virtual bool hasExternalVariable(std::pair<int,int>& pair) override;
            virtual bool hasExternalVariable(const std::string &symbol) override;

            virtual bool hasVariable(int id) override;
            virtual bool hasVariable(int id, int idx) override;
            virtual bool hasVariable(std::pair<int,int>& pair) override;
            virtual bool hasVariable(const std::string &symbol) override;

            virtual Variable *getVariable(int id) override;
            virtual Variable *getVariable(int id, int idx) override;
            virtual Variable *getVariable(std::pair<int,int>& pair) override;
            virtual Variable *getVariable(const std::string &symbol) override;

            virtual std::vector<Variable*> getVariables() override;

            virtual Variable* putVariable(std::pair<int,int>& pair, NDArray *array) override;
            virtual void putVariable(std::pair<int,int>& pair, Variable *variable) override;
            virtual void putVariable(int id, Variable *variable) override;
            virtual void putVariable(int id, NDArray *array) override;
            virtual Variable* putVariable(int id, int idx, NDArray *array) override;
            void putVariable(int id, int idx, const NDArray &array) override;
            virtual void putVariable(int id, int idx, Variable *array) override;

            virtual void replaceVariable(Variable *variable) override;

            virtual void dropVariable(std::pair<int,int> &pair) override;
            virtual void dropVariable(int id, int idx) override;

            virtual void putOutputVariable(Variable *variable) override;

            virtual void trackList(sd::NDArrayList *list) override;

            // memory-related statistics
            virtual Nd4jLong externalMemory() override;
            virtual Nd4jLong internalMemory() override;
            virtual Nd4jLong totalMemory() override;

            virtual int externalEntries() override;
            virtual int internalEntries() override;
            virtual int totalEntries() override;

            virtual VariableSpace *clone() override;

            virtual Stash* getStash() override;
            virtual void setFlowPath(FlowPath* timers) override;
            virtual FlowPath* flowPath() override;
        };
    }
}