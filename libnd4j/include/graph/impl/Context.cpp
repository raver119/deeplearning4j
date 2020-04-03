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

#include <graph/Context.h>
#include <helpers/ShapeUtils.h>
#include <graph/Context.h>
#include <array/InteropDataBuffer.h>


namespace sd {
    namespace graph {
        Context::Context(const ContextPrototype& prototype, VariableSpace* variableSpace, GraphMemoryManager *memoryManager) {
            _memoryManager = memoryManager;
            _variableSpace = variableSpace;

            for (const auto &v: prototype.inputs()) {
                this->_inputs.push_back(v);
            }

            for (const auto &v: prototype.getTArguments()) {
                this->_tArgs.push_back(v);
            }

            for (const auto &v: prototype.getIArguments()) {
                this->_iArgs.push_back(v);
            }

            for (const auto &v: prototype.getBArguments()) {
                this->_bArgs.push_back(v);
            }

            for (const auto &v: prototype.getAxis()) {
                this->_axis.push_back(v);
            }

            this->_opNum = prototype.opNum();
            this->_isInplace = prototype.isInplace();
            this->_nodeId = prototype.nodeId();
            this->_name = prototype.name();
            this->_useMKLDNN = prototype.isUseMKLDNN();
        }

        Context::Context(int nodeId, VariableSpace *variableSpace) {
            this->_nodeId = nodeId;
            this->_variableSpace = variableSpace;
            this->_isInplace = false;
            this->_workspace = nullptr;

            this->_executionTime.first = 0;
            this->_executionTime.second = 0;
        }

        Context::Context(int nodeId, VariableSpace *variableSpace, bool isInplace) : Context(nodeId, variableSpace) {
            this->_isInplace = isInplace;
        }

        Context::~Context() {
            this->_iArgs.clear();
            this->_tArgs.clear();
            this->_inputs.clear();
            this->_fastpath_in.clear();
            this->_fastpath_out.clear();

            if (_context != nullptr)
                delete _context;
        }

        void Context::setTargetEngine(samediff::Engine engine) {
            _engine = engine;
        }

        void Context::attachWorkspace(sd::memory::Workspace* workspace) {
            this->_workspace = workspace;
        }

        void Context::setVariableSpace(VariableSpace *variableSpace) {
            this->_variableSpace = variableSpace;
        }

        const std::vector<std::shared_ptr<NDArray>>& Context::fastpath_in() const {
            return _fastpath_in;
        }

        const std::vector<std::shared_ptr<NDArray>>& Context::fastpath_out() const {
            return _fastpath_out;
        }

        bool Context::isFastPath() const {
            auto ie = _fastpath_in.empty();
            auto io = _fastpath_out.empty();
            // two options here.
            // either both IN/OUT are filled
            auto b1 = (!ie && !io) || (!ie && _isInplace);

            // or at least something is filled, and FastPath is NOT forbidden
            auto b2 = (!ie || !io) && !_forbidFastPath;
            return b1 || b2;
        }

        void Context::forbidFastPath(bool reallyForbid) {
            _forbidFastPath = reallyForbid;
        }

        VariableSpace *Context::getVariableSpace() {
            return _variableSpace;
        }

        sd::memory::Workspace* Context::workspace() const {
            return _workspace;
        }

        Stash* Context::stash() const {
            return _variableSpace->stash();
        }

        Nd4jLong sd::graph::Context::outerTime() const {
            return this->_executionTime.first;
        }

        Nd4jLong sd::graph::Context::innerTime() const {
            return this->_executionTime.second;
        }

        void sd::graph::Context::setOuterTime(Nd4jLong time){
            this->_executionTime.first = time;
        }

        void sd::graph::Context::setInnerTime(Nd4jLong time){
            this->_executionTime.second = time;
        }


        std::shared_ptr<Variable> Context::getVariable(int idx) const {
            if (idx >= this->_inputs.size()) {
                nd4j_printf("Node %i; Variable [%i] requested, but only %i inputs available\n", this->_nodeId, idx, this->_inputs.size());
                throw std::runtime_error("Context: bad Variable index");
            }

            auto p = this->_inputs[idx];

            auto v = variable(p);

            if (Environment::getInstance()->isDebugAndVerbose() && v != nullptr &&  v->getNDArray() != nullptr) {
                auto array = v->getNDArray();
                std::string shape_ = ShapeUtils::shapeAsString(array.get());
                auto type = DataTypeUtils::asString(array->dataType());
                float m = std::numeric_limits<float>::quiet_NaN();
                if (!array->isEmpty()) {
                    auto values = array->asIndexedString(16);

                    nd4j_printf("Debug info for node_%i input[%i]; shape: %s; ews: [%i]; order: [%i]; dtype: [%s]; first values: %s\n", this->_nodeId, idx, shape_.c_str(), array->ews(), array->ordering(), type.c_str(), values.c_str());
                } else {
                    nd4j_printf("Debug info for node_%i input[%i]; shape: %s; ews: [%i]; order: [%i]; dtype: [%s]; mean value: [%f]\n", this->_nodeId, idx, shape_.c_str(), array->ews(), array->ordering(), type.c_str(), m);
                }
            }

            return v;
        }

        std::shared_ptr<Variable> Context::variable(int idx) const {
            return getVariable(idx);
        }

        std::shared_ptr<Variable> Context::variable(std::initializer_list<int> p) const {
            if (p.size() != 2)
                throw std::runtime_error("Variable address should have size of 2");

            // FIXME: lol
            std::vector<int> vec(p);
            std::pair<int, int> pair(vec[0], vec[1]);
            return variable(pair);
        }

        std::shared_ptr<Variable> Context::variable(int node, int idx) const {
            std::pair<int, int> pair(node, idx);
            return variable(pair);
        }

        std::shared_ptr<Variable> Context::variable(const std::pair<int,int>& p) const {
            try {
                return _variableSpace->getVariable(p);
            } catch (std::exception &e) {
                nd4j_printf("Node %i; Non-existent variable requested: [%i:%i]\n", this->_nodeId, p.first, p.second);
                throw std::runtime_error("Bad variable");
            }
        }

        void Context::pushNDArrayToVariableSpace(int nodeId, int index, const NDArray &array) {
            std::pair<int,int> pair(nodeId, index);
            pushNDArrayToVariableSpace(pair, array);
        }

        void Context::pushNDArrayToVariableSpace(const std::pair<int, int> &pair, const NDArray &array) {
            if (_variableSpace != nullptr) {
                if (!_variableSpace->hasVariable(pair)) {
                    auto var = std::make_shared<Variable>(array, "", pair.first, pair.second);
                    _variableSpace->putVariable(pair, var);
                } else {
                    auto var = _variableSpace->getVariable(pair);
                    var->setNDArray(std::make_shared<NDArray>(array));
                }
            }
        }

        void Context::pushNDArrayListToVariableSpace(int nodeId, int index, const NDArrayList &list, bool track) {
            std::pair<int,int> pair(nodeId, index);
            pushNDArrayListToVariableSpace(pair, list, track);
        }

        void Context::pushNDArrayListToVariableSpace(const std::pair<int, int>& pair, const NDArrayList &list, bool track) {
            if (!_variableSpace->hasVariable(pair)) {
                auto var = std::make_shared<Variable>();
                var->setId(pair.first, pair.second);
                var->setNDArrayList(std::make_shared<NDArrayList>(list));
                _variableSpace->putVariable(pair, var);
            } else {
                auto var = _variableSpace->getVariable(pair);
                var->setNDArrayList(std::make_shared<NDArrayList>(list));
            }
        }

        std::shared_ptr<Variable> Context::ensureVariable(const std::string &name, int id, int idx) {
            std::pair<int, int> pair(this->nodeId(), idx);

            if (_variableSpace == nullptr)
                throw std::runtime_error("Context::ensureVariable VariableSpace is NULL!");

            if (!_variableSpace->hasVariable(pair)) {
                auto var = std::make_shared<Variable>();
                var->setId(this->nodeId(), idx);
                auto name = this->name();

                if (!name.empty())
                    var->setName(name);

                _variableSpace->putVariable(pair, var);
                return var;
            } else {
                return _variableSpace->getVariable(pair);
            }
        }

        bool Context::isValueAvailable(const std::string &name, int id, int idx ) const {
            auto var = const_cast<Context*>(this)->ensureVariable(name, id, idx);

            if (var->variableType() == VariableType::NDARRAY) {
                return var->hasNDArray();
            } else if (var->variableType() == VariableType::ARRAY_LIST) {
                return var->hasNDArrayList();
            }

            return false;
        }

        std::shared_ptr<NDArray> Context::getNDArray(int idx) const {
            return array(idx);
        }

        std::shared_ptr<NDArray> Context::array(int idx) const {
            // we check for fastpath first
            if (!_fastpath_in.empty() && _fastpath_in.size() > idx) {
                return _fastpath_in[idx];
            }

            // if no luck for fastpath - return whatever is available
            return getVariable(idx)->getNDArray();
        }

        LaunchContext* Context::launchContext() {
            //FIXME: we need proper context to be shared here
            if (_context == nullptr) {
                return LaunchContext::defaultContext();
            } else {
                return _context;
            }
        }

        unsigned long Context::width() const {
            if (!_fastpath_in.empty())
                return _fastpath_in.size();
            else
                return _inputs.size();
        }

        void Context::setInputArray(int index, const NDArray &array) {
            if (_fastpath_in.size() < index + 1)
                _fastpath_in.resize(index+1);

            _fastpath_in[index] = std::make_shared<NDArray>(array);
        }

        NDArray *Context::arrayForOp(int idx) const {
            auto ptr = array(idx);

            if (ptr.get() != nullptr && ptr->undefined())
                return nullptr;

            return ptr.get();
        }

        void Context::setInputArray(int index, void *buffer, void *shapeInfo, void *specialBuffer, void *specialShapeInfo) {
            auto array = std::make_shared<NDArray>(buffer, specialBuffer, reinterpret_cast<Nd4jLong *>(shapeInfo));

            if (_fastpath_in.size() < index + 1)
                _fastpath_in.resize(index+1);

            _fastpath_in[index] = array;

            if (_context != nullptr)
                array->setContext(_context);
        }

        void Context::setOutputArray(int index, const NDArray &array) {
            if (_fastpath_out.size() < index + 1)
                _fastpath_out.resize(index+1);

            _fastpath_out[index] = std::make_shared<NDArray>(array);
        }

        void Context::setOutputArray(int index, void *buffer, void *shapeInfo, void *specialBuffer, void *specialShapeInfo) {
            if (_fastpath_out.size() < index + 1)
                _fastpath_out.resize(index+1);

            auto array = std::make_shared<NDArray>(buffer, specialBuffer, reinterpret_cast<Nd4jLong *>(shapeInfo));

            _fastpath_out[index] = array;

            if (_context != nullptr)
                array->setContext(_context);
        }

        void Context::setInputArray(int index, void *vdatabuffer, void *shapeInfo, void *specialShapeInfo) {
            auto dataBuffer = reinterpret_cast<InteropDataBuffer*>(vdatabuffer);

            if (_fastpath_in.size() < index + 1)
                _fastpath_in.resize(index+1);

            std::shared_ptr<NDArray> array;
            if (dataBuffer != nullptr)
                array = std::make_shared<NDArray>(dataBuffer->dataBuffer(), reinterpret_cast<Nd4jLong *>(shapeInfo), sd::LaunchContext::defaultContext(), dataBuffer->offset() / DataTypeUtils::sizeOf(ArrayOptions::dataType(reinterpret_cast<Nd4jLong *>(shapeInfo))));
            else
                array = std::make_shared<NDArray>(nullptr, nullptr, reinterpret_cast<Nd4jLong *>(shapeInfo));

            _fastpath_in[index] = array;

            if (_context != nullptr)
                array->setContext(_context);
        }

        void Context::setOutputArray(int index, void *vdatabuffer, void *shapeInfo, void *specialShapeInfo) {
            auto dataBuffer = reinterpret_cast<InteropDataBuffer*>(vdatabuffer);

            if (_fastpath_out.size() < index + 1)
                _fastpath_out.resize(index+1);

            std::shared_ptr<NDArray> array;
            if (dataBuffer != nullptr)
                array = std::make_shared<NDArray>(dataBuffer->dataBuffer(), reinterpret_cast<Nd4jLong *>(shapeInfo), sd::LaunchContext::defaultContext(), dataBuffer->offset() / DataTypeUtils::sizeOf(ArrayOptions::dataType(reinterpret_cast<Nd4jLong *>(shapeInfo))));
            else
                array = std::make_shared<NDArray>(nullptr, nullptr, reinterpret_cast<Nd4jLong *>(shapeInfo));

            _fastpath_out[index] = array;

            if (_context != nullptr)
                array->setContext(_context);
        }

        void Context::setTArguments(double *arguments, int numberOfArguments) {
            _tArgs.clear();
            _tArgs.reserve(numberOfArguments);
            for (int e = 0; e < numberOfArguments; e++)
                _tArgs.push_back(arguments[e]);
        }

        void Context::setIArguments(Nd4jLong *arguments, int numberOfArguments) {
            _iArgs.clear();
            _iArgs.reserve(numberOfArguments);
            for (int e = 0; e < numberOfArguments; e++)
                _iArgs.push_back(arguments[e]);
        }

        void Context::setBArguments(bool *arguments, int numberOfArguments) {
            _bArgs.clear();
            _bArgs.reserve(numberOfArguments);
            for (int e = 0; e < numberOfArguments; e++)
                _bArgs.push_back(arguments[e]);
        }

        void Context::setCudaContext(Nd4jPointer cudaStream, Nd4jPointer reductionPointer, Nd4jPointer allocationPointer) {
#ifdef __CUDABLAS__
            _context = new LaunchContext(cudaStream, reductionPointer, allocationPointer);

            // FIXME: either pass handle from outside, or make sure outside we use the same handle
            _context->setCublasHandle(LaunchContext::defaultContext()->getCublasHandle());

            for (auto v: _fastpath_out)
                v->setContext(_context);

            for (auto v: _fastpath_in)
                v->setContext(_context);
#endif
        }

        void Context::allowHelpers(bool reallyAllow) {
            _helpersAllowed = reallyAllow;
        }

        bool Context::helpersAllowed() const {
            return _helpersAllowed;
        }

        void Context::setTArguments(const std::vector<double> &tArgs) {
            for (auto t:tArgs)
                _tArgs.emplace_back(t);
        }

        void Context::setIArguments(const std::vector<Nd4jLong> &iArgs) {
            for (auto i:iArgs)
                _iArgs.emplace_back(i);
        }

        void Context::setBArguments(const std::vector<bool> &bArgs) {
            for (auto b:bArgs)
                _bArgs.push_back(b);
        }

        void Context::setShapeFunctionOverride(bool reallyOverride) {
            _shapeFunctionOverride = reallyOverride;
        }

        bool Context::shapeFunctionOverride() const {
            return _shapeFunctionOverride;
        }

        samediff::ExecutionMode Context::executionMode() const {
            return _execMode;
        }

        void Context::setExecutionMode(samediff::ExecutionMode executionMode) {
            _execMode = executionMode;
        }

        bool Context::isTraining() const {
            return _execMode == samediff::ExecutionMode::MODE_TRAINING;
        }

        bool Context::isInference() const {
            return _execMode == samediff::ExecutionMode::MODE_INFERENCE;
        }

        void Context::setDArguments(sd::DataType *arguments, int numberOfArguments) {
            _dArgs.clear();
            for (int e = 0; e < numberOfArguments; e++)
                _dArgs.emplace_back(arguments[e]);
        }

        void Context::setDArguments(const std::vector<sd::DataType> &dArgs) {
            _dArgs.clear();
            for (auto d:dArgs)
                _dArgs.emplace_back(d);
        }

        void Context::clearFastPath() {
            _fastpath_in.clear();
            _fastpath_out.clear();
        }

        const GraphMemoryManager &Context::memoryManager() const {
            return *_memoryManager;
        }

        void Context::setInputArray(int index, const std::shared_ptr<NDArray> &array) {
            if (_fastpath_in.size() < index + 1)
                _fastpath_in.resize(index+1);

            _fastpath_in[index] = array;
        }

        void Context::setOutputArray(int index, const std::shared_ptr<NDArray> &array) {
            if (_fastpath_out.size() < index + 1)
                _fastpath_out.resize(index+1);

            _fastpath_out[index] = array;
        }
    }
}

