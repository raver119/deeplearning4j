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

package org.nd4j.autodiff.samediff;
import lombok.Getter;
import org.nd4j.linalg.exception.ND4JIllegalArgumentException;

@Getter
public class SDIndex {

    public enum IndexType{
      ALL,
      POINT,
      INTERVAL
    }

    private IndexType indexType = IndexType.ALL;
    private long pointIndex;
    private boolean pointKeepDim;
    private Long intervalBegin = null;
    private Long intervalEnd = null;
    private Long intervalStrides = 1l;


    public SDIndex(){}

    /**
     * Create an index that gets the entire dimension.
     */
    public static SDIndex all(){
        return new SDIndex();
    }


    /**
     * Create a point index.  The dimension will not be kept.
     * Negative values are supported, and interpreted as being from the end (like Python indexing).
     * @param i The index to get.
     */
    public static SDIndex point(long i){
        SDIndex sdIndex = new SDIndex();
        sdIndex.indexType = IndexType.POINT;
        sdIndex.pointIndex = i;
        sdIndex.pointKeepDim = false;
        return sdIndex;
    }


    /**
     * Create a point index.
     * Negative values are supported, and interpreted as being from the end (like Python indexing).
     * @param i The index to get.
     * @param keepDim Whether to keep the dimension.
     */
    public static SDIndex point(long i, boolean keepDim){
        SDIndex sdIndex = new SDIndex();
        sdIndex.indexType = IndexType.POINT;
        sdIndex.pointIndex = i;
        sdIndex.pointKeepDim = keepDim;
        return sdIndex;
    }

    /**
     * Create an interval index with a stride of 1.
     * Negative values are supported, and interpreted as being from the end (like Python indexing).
     * @param begin The beginning of the interval.
     * @param end The end of the interval (exclusive).
     */
    public static SDIndex interval(Long begin, Long end){
        SDIndex sdIndex = new SDIndex();
        sdIndex.indexType = IndexType.INTERVAL;
        sdIndex.intervalBegin = begin;
        sdIndex.intervalEnd = end;
        return sdIndex;
    }


    /**
     * Create an interval index with a stride of 1.
     * Negative values are supported, and interpreted as being from the end (like Python indexing).
     * @param begin The beginning of the interval.
     * @param end The end of the interval (exclusive).
     */
    public static SDIndex interval(Integer begin, Integer end){
        SDIndex sdIndex = new SDIndex();
        sdIndex.indexType = IndexType.INTERVAL;
        if(begin != null) {
            sdIndex.intervalBegin = begin.longValue();
        }
        if(end != null){
            sdIndex.intervalEnd = end.longValue();
        }
        return sdIndex;
    }


    /**
     * Create an interval index.
     * Negative endpoints are supported, and interpreted as being from the end (like Python indexing).
     * @param begin The beginning of the interval.
     * @param strides The stride of the interval.
     * @param end The end of the interval (exclusive).
     */
    public static SDIndex interval(Long begin, Long strides, Long end){
        if(strides == 0){
            throw new ND4JIllegalArgumentException("Invalid index : strides can not be 0.");
        }
        SDIndex sdIndex = new SDIndex();
        sdIndex.indexType = IndexType.INTERVAL;
        sdIndex.intervalBegin = begin;
        sdIndex.intervalEnd = end;
        sdIndex.intervalStrides = strides;
        return sdIndex;
    }

    /**
     * Create an interval index.
     * Negative endpoints are supported, and interpreted as being from the end (like Python indexing).
     * @param begin The beginning of the interval.
     * @param strides The stride of the interval.
     * @param end The end of the interval (exclusive).
     */
    public static SDIndex interval(Integer begin, Integer strides, Integer end){
        if(strides == 0){
            throw new ND4JIllegalArgumentException("Invalid index : strides can not be 0.");
        }
        SDIndex sdIndex = new SDIndex();
        sdIndex.indexType = IndexType.INTERVAL;
        if(begin != null) {
            sdIndex.intervalBegin = begin.longValue();
        }
        if(end != null){
            sdIndex.intervalEnd = end.longValue();
        }
        if(strides != null){
            sdIndex.intervalStrides = strides.longValue();
        }
        return sdIndex;
    }
    
}
