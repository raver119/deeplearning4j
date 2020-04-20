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
// This class describes collection of NDArrays
//
// @author raver119!gmail.com
//

#ifndef NDARRAY_LIST_H
#define NDARRAY_LIST_H

#include <string>
#include <atomic>
#include <unordered_map>
#include <array/NDArray.h>
#include <memory/Workspace.h>
#include <system/dll.h>

namespace sd {
    class SD_EXPORT NDArrayList {
    protected:
        class InternalArrayList {
        public:
            // numeric and symbolic ids of this list
            std::pair<int, int> _id;
            std::string _name;

            sd::DataType _dtype;

            // stored chunks
            MAP_IMPL<int, sd::NDArray> _chunks;

            // just a counter, for stored elements
            std::atomic<int> _elements;
            mutable std::atomic<int> _counter;

            // reference shape
            std::vector<Nd4jLong> _shape;

            // unstack axis
            int _axis = 0;

            //
            bool _expandable = false;

            // maximum number of elements
            int _height = 0;


            //////////
            InternalArrayList(int height = 0, bool expandable = false);
            ~InternalArrayList() = default;
        };

        std::shared_ptr<InternalArrayList> _state;

    public:
        NDArrayList(int height = 0, bool expandable = false);
        ~NDArrayList();

        NDArrayList(const sd::NDArrayList &other);
        NDArrayList(sd::NDArrayList &&other);

        NDArrayList& operator=(const NDArrayList& other) noexcept;

        // move assignment operator
        NDArrayList& operator=(NDArrayList&& other) noexcept;

        sd::DataType dataType() const;

        NDArray read(int idx);
        NDArray readRaw(int idx);
        Nd4jStatus write(int idx, const NDArray &array);

        NDArray pick(const std::vector<int>& indices);
        bool isWritten(int index) const;

        const std::vector<Nd4jLong>& shape() const;
        void setShape(const std::vector<Nd4jLong> &shape);

        NDArray stack() const;
        void unstack(const NDArray &array, int axis);

        const std::pair<int,int>& id() const;
        const std::string& name() const;

        NDArrayList clone();

        bool equals(NDArrayList& other);

        int elements() const;
        int height() const;

        int counter() const;
    };
}

#endif