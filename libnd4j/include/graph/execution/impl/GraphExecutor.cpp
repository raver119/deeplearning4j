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

#include <graph/execution/GraphExecutor.h>
#include <graph/Graph.h>

namespace sd {
    namespace graph {
        Context GraphExecutor::prepareContext(ContextPrototype *contextPrototype, VariableSpace &variableSpace, const GraphMemoryManager &memoryManager) const {
            // TODO: maybe we'll want to do something here?
            return Context(*contextPrototype, &variableSpace, const_cast<GraphMemoryManager*>(&memoryManager));
        }

        Nd4jStatus GraphExecutor::preprocess(sd::ops::DeclarableOp *op, Context &context) const {
            // time to allocate outputs, if that's not inplace op
            // inplace case is covered there
            op->prepareOutputs(context);

            // once prepareOutputs method was called - we don't need shape function anymore
            context.setShapeFunctionOverride(true);

            return Status::OK();
        }

        Nd4jStatus GraphExecutor::postprocess(sd::ops::DeclarableOp *op, Context *context) const {
            return Status::OK();
        }


        Nd4jStatus GraphExecutor::execute(sd::ops::DeclarableOp *op, ContextPrototype *contextPrototype, const OpSequence &sequence, const OptimizedGraph &graph, const int deviceId) const {
            auto ctx = prepareContext(contextPrototype, *graph.originalGraph().getVariableSpace(), graph.memoryManager());
            return op->execute(&ctx);
        }

        Nd4jStatus GraphExecutor::execute(const OpSequence &sequence, const OptimizedGraph &graph, const int deviceId) const {
            /*
             * this is a basic implementation that works without dispatching etc
             */
            for (int e = 0; e < sequence.length(); e++) {
                auto v = sequence[e];
                auto result = execute(v.first, v.second, sequence, graph, deviceId >= 0 ? deviceId : sequence.deviceId());
                if (result != Status::OK())
                    return result;
            }

            return Status::OK();
        }

        Nd4jStatus GraphExecutor::execute(const OptimizedGraph &graph) const {
            const auto numDevices = AffinityManager::numberOfDevices();

            /*
             * this is a basic exection logic: roll through layers and sequences and execute them one by one sequentially
             */
            Nd4jStatus result = Status::OK();
            for (uint64_t l = 0; l < graph.layers(); l++) {
                auto layer = graph.layer(l);

                for (uint64_t o = 0; layer.width(); o++) {
                    execute(layer[o], graph);
                }

                // optionally block until all sequences in this layer processed
                if (layer.width() > 0 && numDevices > 1)
                    for (uint64_t o = 0; layer.width(); o++) {
                        result = layer[o].wait();
                        if (result != Status::OK())
                            return result;
                    }
            }

            return result;
        }
    }
}
