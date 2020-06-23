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

#ifndef SD_EXECUTIONLAYER_H
#define SD_EXECUTIONLAYER_H

#include <graph/execution/OpSequence.h>
#include <system/dll.h>

#include <vector>

namespace sd {
namespace graph {
class SD_EXPORT ExecutionLayer {
 protected:
  std::vector<OpSequence> _sequences;

 public:
  ExecutionLayer(const std::vector<OpSequence>& sequences = {});
  ~ExecutionLayer() = default;

  ExecutionLayer(const ExecutionLayer& other) noexcept;

  ExecutionLayer& operator=(const ExecutionLayer& other) noexcept;

  // move constructor
  ExecutionLayer(ExecutionLayer&& other) noexcept;

  // move assignment operator
  ExecutionLayer& operator=(ExecutionLayer&& other) noexcept;

  /**
   * This method returns number of sequences in this layer
   * @return
   */
  uint64_t width() const;

  /**
   * This method returns specified OpSequence from this layer
   * @return
   */
  const OpSequence& at(uint64_t index) const;
  const OpSequence& operator[](uint64_t index) const;

  /**
   * This method appends OpSequence to the end of this layer
   * @param sequence
   */
  void append(OpSequence&& sequence);
  void append(const OpSequence& sequence);

  /**
   * sort OpSequences in increasing order in respect to id of fist node in sequence
   * @param sequence
   */
  void sortOpSequences();

  /**
   * This method checks if specified Node resides within this ExecutionLayer
   * @param nodeId
   * @return
   */
  bool hasNode(int nodeId) const;

  /**
   * This method removes all empty OpSequences from this layer
   */
  void purgeEmptySequences();
};

}  // namespace graph
}  // namespace sd

#endif  // SD_EXECUTIONLAYER_H
