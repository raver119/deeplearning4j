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

#ifndef SD_BROADCASTINTOPSTUPLE_H
#define SD_BROADCASTINTOPSTUPLE_H

#include <system/dll.h>
#include <system/op_enums.h>

namespace sd {
class SD_EXPORT BroadcastIntOpsTuple {
 private:
 public:
  sd::scalar::IntOps s;
  sd::pairwise::IntOps p;
  sd::broadcast::IntOps b;

  BroadcastIntOpsTuple() = default;
  ~BroadcastIntOpsTuple() = default;

  BroadcastIntOpsTuple(sd::scalar::IntOps scalar, sd::pairwise::IntOps pairwise,
                       sd::broadcast::IntOps broadcast) {
    s = scalar;
    p = pairwise;
    b = broadcast;
  }

  static BroadcastIntOpsTuple custom(sd::scalar::IntOps scalar,
                                     sd::pairwise::IntOps pairwise,
                                     sd::broadcast::IntOps broadcast);
};
}  // namespace sd

#endif  // SD_BROADCASTOPSTUPLE_H
