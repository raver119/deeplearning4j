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

#ifndef SD_FILEUTILS_H
#define SD_FILEUTILS_H

#include <cstdint>
#include <system/dll.h>

namespace sd {
    class SD_EXPORT FileUtils {
    public:
        static bool fileExists(const char *filename);

        static int64_t fileSize(const char *filename);
    };
}


#endif //SD_FILEUTILS_H
