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
// Created by raver119 on 2018-09-16.
// @author Oleg Semeniv <oleg.semeniv@gmail.com>
// @author GS (shugeo) <sgazeos@gmail.com>
//

#ifndef DEV_TESTS_NDARRAYFACTORY_H
#define DEV_TESTS_NDARRAYFACTORY_H

#include <vector>
#include <array/Order.h>
#include <array/NDArray.h>
//#include <memory/Workspace.h>
#include <execution/LaunchContext.h>
#include <string>


namespace sd {
    class ND4J_EXPORT NDArrayFactory {
    private:
        template <typename T>
        static void memcpyFromVector(void *ptr, const std::vector<T> &vector);
    public:
        template <typename T>
        static NDArray empty(sd::LaunchContext * context = sd::LaunchContext ::defaultContext());

        static NDArray empty(sd::DataType dataType, sd::LaunchContext * context = sd::LaunchContext ::defaultContext());

        template <typename T>
        static NDArray valueOf(const std::vector<Nd4jLong>& shape, T value, const Order order = sd::kArrayOrderC,  sd::LaunchContext * context = sd::LaunchContext ::defaultContext());

        static NDArray valueOf(const std::vector<Nd4jLong>& shape, const NDArray& value, const Order order = sd::kArrayOrderC, sd::LaunchContext* context = sd::LaunchContext ::defaultContext());

        template <typename T>
        static NDArray linspace(T from, T to, Nd4jLong numElements);

        template <typename T>
        static NDArray create(const T value, sd::LaunchContext * context = sd::LaunchContext ::defaultContext());

        static NDArray create(sd::DataType dtype, sd::LaunchContext * context = sd::LaunchContext ::defaultContext());

        template <typename T>
        static NDArray create(DataType type, const T scalar, sd::LaunchContext * context = sd::LaunchContext ::defaultContext());

        template <typename T>
        static NDArray vector(Nd4jLong length, T startingValue = (T) 0, sd::LaunchContext * context = sd::LaunchContext ::defaultContext());

        template <typename T>
        static NDArray create(const std::vector<Nd4jLong> &shape, const std::vector<T> &data = {}, const sd::Order order = sd::kArrayOrderC, sd::LaunchContext* context = sd::LaunchContext::defaultContext());

        static NDArray create(sd::DataType dtype, const std::vector<Nd4jLong> &shape, const sd::Order order = kArrayOrderC, sd::LaunchContext * context = sd::LaunchContext ::defaultContext());

        template <typename T>
        static NDArray vector(const std::vector<T> &values, sd::LaunchContext* context = sd::LaunchContext ::defaultContext());

#ifndef __JAVACPP_HACK__
        /**
         * This method creates NDArray from .npy file
         * @param fileName
         * @return
         */
        static NDArray fromNpyFile(const char *fileName);

        /**
         * This factory create array from utf8 string
         * @return NDArray default dataType UTF8
         */
        static NDArray string(const char *string, sd::DataType dtype = sd::DataType::UTF8, sd::LaunchContext * context = sd::LaunchContext ::defaultContext());
        static NDArray string(const std::string& string, sd::DataType dtype = sd::DataType::UTF8, sd::LaunchContext* context = sd::LaunchContext::defaultContext());

        /**
         * This factory create array from utf16 string
         * @return NDArray default dataType UTF16
         */
        static NDArray string(const char16_t* u16string, sd::DataType dtype = sd::DataType::UTF16, sd::LaunchContext* context = sd::LaunchContext::defaultContext());
        static NDArray string(const std::u16string& u16string, sd::DataType dtype = sd::DataType::UTF16, sd::LaunchContext* context = sd::LaunchContext::defaultContext());
        
        /**
         * This factory create array from utf32 string
         * @return NDArray default dataType UTF32
         */
        static NDArray string(const char32_t* u32string, sd::DataType dtype = sd::DataType::UTF32, sd::LaunchContext* context = sd::LaunchContext::defaultContext());
        static NDArray string(const std::u32string& u32string, sd::DataType dtype = sd::DataType::UTF32, sd::LaunchContext* context = sd::LaunchContext::defaultContext());

        /**
         * This factory create array from vector of utf8 strings
         * @return NDArray default dataType UTF8
         */
        static NDArray string( const std::vector<Nd4jLong> &shape, const std::vector<const char *> &strings, sd::DataType dtype = sd::DataType::UTF8, sd::LaunchContext * context = sd::LaunchContext ::defaultContext());
        static NDArray string( const std::vector<Nd4jLong> &shape, const std::vector<std::string> &string, sd::DataType dtype = sd::DataType::UTF8, sd::LaunchContext * context = sd::LaunchContext ::defaultContext());

        /**
         * This factory create array from vector of utf16 strings
         * @return NDArray default dataType UTF16
         */
        static NDArray string( const std::vector<Nd4jLong>& shape, const std::vector<const char16_t*>& strings, sd::DataType dtype = sd::DataType::UTF16, sd::LaunchContext* context = sd::LaunchContext::defaultContext());
        static NDArray string( const std::vector<Nd4jLong>& shape, const std::vector<std::u16string>& string, sd::DataType dtype = sd::DataType::UTF16, sd::LaunchContext* context = sd::LaunchContext::defaultContext());

        /**
         * This factory create array from vector of utf32 strings
         * @return NDArray default dataType UTF32
         */
        static NDArray string( const std::vector<Nd4jLong>& shape, const std::vector<const char32_t*>& strings, sd::DataType dtype = sd::DataType::UTF32, sd::LaunchContext* context = sd::LaunchContext::defaultContext());
        static NDArray string( const std::vector<Nd4jLong>& shape, const std::vector<std::u32string>& string, sd::DataType dtype = sd::DataType::UTF32, sd::LaunchContext* context = sd::LaunchContext::defaultContext());


        static ResultSet createSetOfArrs(const Nd4jLong numOfArrs, const void* buffer, const Nd4jLong* shapeInfo, const Nd4jLong* offsets, sd::LaunchContext * context = sd::LaunchContext ::defaultContext());

#endif
    };
}

#endif //DEV_TESTS_NDARRAYFACTORY_H
