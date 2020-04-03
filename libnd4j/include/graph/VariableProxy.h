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

            virtual int numberOfPlaceholders() const override;
            virtual const std::vector<std::shared_ptr<Variable>>& placeholders() const override;

            virtual bool hasExternalVariable(int it) const override;
            virtual bool hasExternalVariable(const std::pair<int,int>& pair) const override;
            virtual bool hasExternalVariable(const std::string &symbol) const override;

            virtual bool hasVariable(int id) const override;
            virtual bool hasVariable(int id, int idx) const override;
            virtual bool hasVariable(const std::pair<int,int>& pair) const override;
            virtual bool hasVariable(const std::string &symbol) const override;

            virtual std::shared_ptr<Variable> getVariable(int id) const override;
            virtual std::shared_ptr<Variable> getVariable(int id, int idx) const override;
            virtual std::shared_ptr<Variable> getVariable(const std::pair<int,int>& pair) const override;
            virtual std::shared_ptr<Variable> getVariable(const std::string &symbol) const override;

            virtual std::vector<std::shared_ptr<Variable>> variables() const override;

            virtual std::shared_ptr<Variable> putVariable(const std::pair<int,int>& pair, const NDArray &array) override;
            virtual std::shared_ptr<Variable> putVariable(int id, const NDArray &array) override;
            virtual std::shared_ptr<Variable> putVariable(int id, int idx, const NDArray &array) override;
            virtual std::shared_ptr<Variable> putVariable(const std::string &name, int id, int idx, const NDArray &array) override;
            virtual void putVariable(const std::string& name, int id, int idx, const std::shared_ptr<Variable> &variable) override;
            virtual void putVariable(const std::pair<int,int>& pair, const std::shared_ptr<Variable> &variable) override;
            virtual void putVariable(int id, const std::shared_ptr<Variable> &variable) override;

            virtual void replaceVariable(std::shared_ptr<Variable> variable) override;

            virtual void dropVariable(const std::pair<int,int> &pair) override;
            virtual void dropVariable(int id, int idx) override;

            virtual void putOutputVariable(std::shared_ptr<Variable> variable) override;

            // memory-related statistics
            virtual Nd4jLong externalMemory() const override;
            virtual Nd4jLong internalMemory() const override;
            virtual Nd4jLong totalMemory() const override;

            virtual int externalEntries() const override;
            virtual int internalEntries() const override;
            virtual int totalEntries() const override;


            virtual Stash* stash() const override;
        };
    }
}