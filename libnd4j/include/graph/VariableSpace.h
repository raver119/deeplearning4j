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

#ifndef LIBND4J_VARIABLESPACE_H
#define LIBND4J_VARIABLESPACE_H

#include <helpers/logger.h>
#include <helpers/helper_random.h>
#include <string>
#include <vector>
#include <list>
#include <unordered_map>
#include <mutex>
#include <array/NDArray.h>
#include <array/NDArrayList.h>
#include <graph/Variable.h>
#include <memory/Workspace.h>
#include <graph/Stash.h>
#include <graph/FlowPath.h>


namespace sd {
    namespace graph {
        class SD_EXPORT VariableSpace {
        protected:
            // stash is NOT cloned
            Stash _stash;

            // lookup tables: by name, by id, by id:idx
            MAP_IMPL<std::pair<int, int>, std::shared_ptr<Variable>> _paired;
            MAP_IMPL<std::string, std::shared_ptr<Variable>> _symbolic;
            MAP_IMPL<int, std::shared_ptr<Variable>> _variables;

            // direct references to external variables and internally-generated variables
            std::vector<std::shared_ptr<Variable>> _external;
            std::vector<std::shared_ptr<Variable>> _internal;

            // meh
            std::vector<std::shared_ptr<sd::NDArrayList>> _lists;

            // placeholders. must be resolved before Graph execution
            std::vector<std::shared_ptr<Variable>> _placeholders;

            void silentPutVariable(const std::pair<int,int>& pair, const std::shared_ptr<Variable> &variable);

            int _auto_counter = -1;

            std::mutex _varmap;

        public:
            VariableSpace();
            virtual ~VariableSpace();

            VariableSpace(const sd::graph::VariableSpace &variableSpace);
            VariableSpace(sd::graph::VariableSpace &&variableSpace);

            virtual VariableSpace& operator=(const VariableSpace& other);
            virtual VariableSpace& operator=(VariableSpace&& other);

            virtual int numberOfPlaceholders() const;

            virtual const std::vector<std::shared_ptr<Variable>>& placeholders() const;

            virtual bool hasExternalVariable(int it) const;
            virtual bool hasExternalVariable(const std::pair<int,int>& pair) const;
            virtual bool hasExternalVariable(const std::string &symbol) const;

            virtual bool hasVariable(int id) const;
            virtual bool hasVariable(int id, int idx) const;
            virtual bool hasVariable(const std::pair<int,int>& pair) const;
            virtual bool hasVariable(const std::string &symbol) const;

            virtual std::shared_ptr<Variable> getVariable(int id) const;
            virtual std::shared_ptr<Variable> getVariable(int id, int idx) const;
            virtual std::shared_ptr<Variable> getVariable(const std::pair<int,int>& pair) const;
            virtual std::shared_ptr<Variable> getVariable(const std::string &symbol) const;

            virtual std::vector<std::shared_ptr<Variable>> variables() const;

            virtual std::shared_ptr<Variable> putVariable(const std::pair<int,int>& pair, const NDArray &array);
            virtual std::shared_ptr<Variable> putVariable(int id, const NDArray &array);
            virtual std::shared_ptr<Variable> putVariable(int id, int idx, const std::shared_ptr<NDArray> &array);
            virtual std::shared_ptr<Variable> putVariable(int id, int idx, const NDArray &array);
            virtual std::shared_ptr<Variable> putVariable(const std::string &name, int id, int idx, const NDArray &array);
            virtual void putVariable(const std::string& name, int id, int idx, const std::shared_ptr<Variable> &variable);
            virtual void putVariable(const std::pair<int,int>& pair, const std::shared_ptr<Variable> &variable);
            virtual void putVariable(int id, const std::shared_ptr<Variable> &variable);

            virtual void dropVariable(const std::string &pair);
            virtual void dropVariable(const std::pair<int,int> &pair);
            virtual void dropVariable(int id, int idx);

            virtual void putOutputVariable(std::shared_ptr<Variable> variable);

            virtual void replaceVariable(std::shared_ptr<Variable> variable);

            // memory-related statistics
            virtual Nd4jLong externalMemory() const;
            virtual Nd4jLong internalMemory() const;
            virtual Nd4jLong totalMemory() const;

            virtual int externalEntries() const;
            virtual int internalEntries() const;
            virtual int totalEntries() const;

            void injectVariable(const std::pair<int, int> &pair, std::shared_ptr<Variable> variable);

            virtual Stash* stash() const;

        };
    }
}


#endif //LIBND4J_VARIABLESPACE_H
