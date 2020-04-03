/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 * Copyright (c) 2019-2020 Konduit K.K.
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

#ifndef LIBND4J_CONTEXT_H
#define LIBND4J_CONTEXT_H

#include <vector>
#include <array/NDArray.h>
#include <graph/Variable.h>
#include <graph/VariableSpace.h>
#include <graph/ContextPrototype.h>
#include <memory/GraphMemoryManager.h>
#include <memory/Workspace.h>
#include <execution/Engine.h>

// CUDA-specific includes
#ifdef __CUDACC__

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#endif

namespace sd {
    namespace graph {
        /**
         * This class defines input desired for any given node/operation within graph
         */
        class SD_EXPORT Context : public sd::graph::ContextPrototype {
        protected:
            sd::graph::GraphMemoryManager *_memoryManager = nullptr;
            sd::memory::Workspace* _workspace = nullptr;

            sd::graph::VariableSpace* _variableSpace = nullptr;
            std::pair<Nd4jLong, Nd4jLong> _executionTime;

            sd::DataType _dataType = sd::DataType::FLOAT32;
            // branch for divergent_op
            int _branch = 0;

            // temporary context for standalone ops execution
            LaunchContext* _context = nullptr;

            std::vector<sd::DataType> _dataTypes;

            // fields for fast execution (out-of-graph ops use)
            std::vector<std::shared_ptr<NDArray>> _fastpath_in;
            std::vector<std::shared_ptr<NDArray>> _fastpath_out;

            bool _helpersAllowed = true;

            // in some cases we might be able to skip shape function for validation purposes
            bool _shapeFunctionOverride = false;

            // special flag used during conversion from Graph exec to FastPath exec
            bool _forbidFastPath = false;
        public:
            Context(const ContextPrototype &prototype, VariableSpace* variableSpace, GraphMemoryManager *memoryManager = nullptr);

            explicit Context(int nodeId, VariableSpace *variableSpace = nullptr);
            Context(int nodeId, VariableSpace *variableSpace, bool isInplace);

            // default destructor
            ~Context();

            // these methods are for execution timing
            void setOuterTime(Nd4jLong time);
            void setInnerTime(Nd4jLong time);
            Nd4jLong outerTime() const;
            Nd4jLong innerTime() const;

            // these methods are related to Workspace abstraction
            bool hasWorkspaceProvided() const;
            void attachWorkspace(sd::memory::Workspace* workspace);

            sd::memory::Workspace* workspace() const;


            void setVariableSpace(VariableSpace* variableSpace);

            void setTargetEngine(samediff::Engine engine);

            VariableSpace *getVariableSpace();

            LaunchContext* launchContext();

            /**
             *
             * @return
             */
            Stash* stash() const;

            /**
             * This method returns variable for a given input index for this block
             * @param idx
             * @return
             */
            std::shared_ptr<Variable> getVariable(int idx) const;
            std::shared_ptr<Variable> variable(int idx) const;

            /**
             * This method is shortcut to getVariable(int idx);
             *
             * + it check fastpath for array availability (preferred)
             * @return
             */
            std::shared_ptr<NDArray> getNDArray(int idx) const;
            std::shared_ptr<NDArray> array(int idx) const;

            /**
             * This is special method, used only within Graph
             * @param idx
             * @return
             */
            NDArray* arrayForOp(int idx) const;


            /**
             * This method fetches variable from VariableSpace DIRECTLY
             * @param p
             * @return
             */
            std::shared_ptr<Variable> variable(int node, int index) const;
            std::shared_ptr<Variable> variable(const std::pair<int,int>& p) const;
            std::shared_ptr<Variable> variable(std::initializer_list<int> p) const;


            void pushNDArrayToVariableSpace(int nodeId, int index, const NDArray &array);
            void pushNDArrayToVariableSpace(const std::pair<int, int>& pair, const NDArray &array);

            void pushNDArrayListToVariableSpace(int nodeId, int index, std::shared_ptr<NDArrayList> list);
            void pushNDArrayListToVariableSpace(int nodeId, int index, const NDArrayList &list, bool track = true);
            void pushNDArrayListToVariableSpace(const std::pair<int, int>& pair, const NDArrayList &list, bool track = true);

            bool isValueAvailable(const std::string &name, int id, int idx = 0) const;

            std::shared_ptr<Variable> ensureVariable(const std::string &name, int id, int idx = 0);

            unsigned long width() const override;

            // methods used in java interop
            /**
             * This method checks if Context uses fastpath variable access
             * @return
             */
            bool isFastPath() const;

            /**
             * Method allows to forbid FastPath execution
             * @param reallyForbid
             */
            void forbidFastPath(bool reallyForbid);

#ifndef __JAVACPP_HACK__
            const std::vector<std::shared_ptr<NDArray>>& fastpath_in() const;
            const std::vector<std::shared_ptr<NDArray>>& fastpath_out() const;
#endif

            void setInputArray(int index, const std::shared_ptr<NDArray> &array);
            void setInputArray(int index, const NDArray &array);
            void setInputArray(int index, void *buffer, void *shapeInfo, void *specialBuffer, void *specialShapeInfo);
            void setInputArray(int index, void *databuffer, void *shapeInfo, void *specialShapeInfo);

            void setOutputArray(int index, const std::shared_ptr<NDArray> &array);
            void setOutputArray(int index, const NDArray &array);
            void setOutputArray(int index, void *buffer, void *shapeInfo, void *specialBuffer, void *specialShapeInfo);
            void setOutputArray(int index, void *databuffer, void *shapeInfo, void *specialShapeInfo);

            void setTArguments(double *arguments, int numberOfArguments);
            void setIArguments(Nd4jLong *arguments, int numberOfArguments);
            void setBArguments(bool *arguments, int numberOfArguments);
            void setDArguments(sd::DataType *arguments, int numberOfArguments);

            void setTArguments(const std::vector<double> &tArgs);
            void setIArguments(const std::vector<Nd4jLong> &tArgs);
            void setBArguments(const std::vector<bool> &tArgs);
            void setDArguments(const std::vector<sd::DataType> &dArgs);

            /**
             * This method purges fastpath in/out contents and releases all the handles.
             *
             * PLEASE NOTE: I/T/B/D args will stay intact
             */
            void clearFastPath();

            void setCudaContext(Nd4jPointer cudaStream, Nd4jPointer reductionPointer, Nd4jPointer allocationPointer);

            void allowHelpers(bool reallyAllow);
            bool helpersAllowed() const;

            void setShapeFunctionOverride(bool reallyOverride);
            bool shapeFunctionOverride() const;

            samediff::ExecutionMode executionMode() const;
            void setExecutionMode(samediff::ExecutionMode executionMode);

            bool isTraining() const;
            bool isInference() const;

            const GraphMemoryManager& memoryManager() const;
        };
    }
}


#endif //LIBND4J_BLOCK_H
