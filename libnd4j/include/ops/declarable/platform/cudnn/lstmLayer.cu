/*******************************************************************************
 * Copyright (c) 2019 Konduit K.K.
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
// @author Yurii Shyrma (iuriish@yahoo.com)
//


#include "cudnnUtils.h"
#include <ops/declarable/helpers/convolutions.h>

namespace nd4j      {
namespace ops       {
namespace platforms {

//////////////////////////////////////////////////////////////////////////
static void lstmLayerCUDNN() {


}

//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(lstmLayer, ENGINE_CUDA) {

     // it  = σ(Wxi * xt  +  Wri * ht-1  +  bi)
    // ft  = σ(Wxf * xt  +  Wrf * ht-1  +  bf)
    // c't = tanh(Wxc * xt  +  Wrc * ht-1  +  bc)
    // ct  = ft ◦ ct-1 + it ◦ c't
    // ot  = σ(Wxo * xt  +  Wro * ht-1  +  bo)
    // ht  = ot ◦ tanh(ct)

    // equations (peephole connections are present)
    // it  = σ(Wxi * xt  +  Wri * ht-1  +  Wpi ◦ ct-1  +  bi)
    // ft  = σ(Wxf * xt  +  Wrf * ht-1  +  Wpf ◦ ct-1  +  bf)
    // c't = tanh(Wxc * xt  +  Wrc * ht-1  +  bc)
    // ct  = ft ◦ ct-1 + it ◦ c't
    // ot  = σ(Wxo * xt  +  Wro * ht-1  +  Wpo ◦ ct  +  bo)
    // ht  = ot ◦ tanh(ct)

    // notations:
    // bS - batch size
    // sL - sequence length, number of time steps
    // nIn - input size
    // nOut - output size (hidden size)

    //     INPUTS:

    // *******
    // input x:
    // 1) [sL, bS, nIn]  when dataFormat == 0
    // 2) [bS, sL, nIn]  when dataFormat == 1
    // 3) [bS, nIn, sL]  when dataFormat == 2

    // *******
    // input weights Wx:
    // 1) [nIn, 4*nOut]    when directionMode <  2
    // 2) [2, nIn, 4*nOut] when directionMode >= 2

    // *******
    // recurrent weights Wr:
    // 1) [nOut, 4*nOut]    when directionMode <  2
    // 2) [2, nOut, 4*nOut] when directionMode >= 2

    // *******
    // peephole weights Wp:
    // 1) [3*nOut]    when directionMode <  2
    // 2) [2, 3*nOut] when directionMode >= 2

    // *******
    // biases b:
    // 1) [4*nOut]    when directionMode <  2
    // 2) [2, 4*nOut] when directionMode >= 2

    // *******
    // sequence length array seqLen:
    // 1) [bS] always

    // *******
    // initial output hI:
    // 1) [bS, nOut]    when directionMode <  2
    // 2) [2, bS, nOut] when directionMode >= 2

    // *******
    // initial cell state cI (same shape as in hI):
    // 1) [bS, nOut]    when directionMode <  2
    // 2) [2, bS, nOut] when directionMode >= 2


    //     OUTPUTS:

    // *******
    // output h:
    // 1) [sL, bS, nOut]    when directionMode <= 2 && dataFormat == 0
    // 2) [bS, sL, nOut]    when directionMode <= 2 && dataFormat == 1
    // 3) [bS, nOut, sL]    when directionMode <= 2 && dataFormat == 2
    // 4) [sL, bS, 2*nOut]  when directionMode == 3 && dataFormat == 0
    // 5) [bS, sL, 2*nOut]  when directionMode == 3 && dataFormat == 1
    // 6) [bS, 2*nOut, sL]  when directionMode == 3 && dataFormat == 2
    // 7) [sL, 2, bS, nOut] when directionMode == 4 && dataFormat == 3

    // *******
    // output at last step hL:
    // 1) [bS, nOut]    when directionMode <  2
    // 2) [2, bS, nOut] when directionMode >= 2

    // *******
    // cell state at last step cL (same shape as in hL):
    // 1) [bS, nOut]    when directionMode <  2
    // 2) [2, bS, nOut] when directionMode >= 2

    // !!! dimension 4*nOut implies order it, ft, c't, ot
    // !!! dimension 3*nOut implies order it, ft, ot

}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(lstmLayer, ENGINE_CUDA) {

    const auto x = INPUT_VARIABLE(0);
    const auto Wx = INPUT_VARIABLE(1);
    const auto Wr = INPUT_VARIABLE(2);

    const auto dataFormat    = INT_ARG(0);    // for unidirectional: 0 = [sL, bS, nIn], 1 = [bS, sL ,nIn], 2 = [bS, nIn, sL], for bidirectional: 3 = [sL, 2, bS, nOut] (ONNX)
    const auto directionMode = INT_ARG(1);    // direction: 0 = fwd, 1 = bwd, 2 = bidirectional sum, 3 = bidirectional concat, 4 = bidirectional extra output dim (in conjunction with format dataFormat = 3)

    const auto hasBiases  = B_ARG(0);   // indicates whether biases array is provided
    const auto hasSeqLen  = B_ARG(1);   // indicates whether seqLen array is provided
    const auto hasInitH   = B_ARG(2);   // indicates whether initial output is provided
    const auto hasInitC   = B_ARG(3);   // indicates whether initial cell state is provided
    const auto hasPH      = B_ARG(4);   // indicates whether peephole connections are present
    const auto retFullSeq = B_ARG(5);   // indicates whether to return whole time sequence h {h_0, h_1, ... , h_sL-1}
    const auto retLastH   = B_ARG(6);   // indicates whether to return output at last time step only, in this case shape would be [bS, nOut] (exact shape depends on dataFormat argument)
    const auto retLastC   = B_ARG(7);   // indicates whether to return cells state at last time step only, in this case shape would be [bS, nOut] (exact shape depends on dataFormat argument)

    if(directionMode == 2 || directionMode == 4)
        return false;

    if(hasPH)
        return false;

    uint count = 3;
    const auto b      = hasBiases ? INPUT_VARIABLE(count++) : nullptr;  // biases
    const auto seqLen = hasSeqLen ? INPUT_VARIABLE(count++) : nullptr;  // seqLen vector
    const auto hI     = hasInitH  ? INPUT_VARIABLE(count++) : nullptr;  // initial output
    const auto cI     = hasInitC  ? INPUT_VARIABLE(count++) : nullptr;  // initial cell state
    const auto Wp     = hasPH     ? INPUT_VARIABLE(count++) : nullptr;  // peephole weights

    count = 0;
    auto h  = retFullSeq ? OUTPUT_VARIABLE(count++) : nullptr;           // output
    auto hL = retLastH   ? OUTPUT_VARIABLE(count++) : nullptr;           // output at last step
    auto cL = retLastC   ? OUTPUT_VARIABLE(count++) : nullptr;           // cell state at last step

    if(hasInitH && hI->ews() != 1)
        return false;

    if(hasInitC && cI->ews() != 1)
        return false;

    if(retFullSeq && h->ews() != 1)
        return false;

    if(retLastH && hL->ews() != 1)
        return false;

    if(retLastC && cL->ews() != 1)
        return false;

    const auto xDataType = x->dataType();

    bool goodType = xDataType == DataType::DOUBLE || xDataType == DataType::FLOAT32 || xDataType != DataType::HALF;
    goodType &= Wx->dataType() == xDataType && Wr->dataType() == xDataType;
    if(hasBiases)
        goodType &= b->dataType() == xDataType;
    if(hasSeqLen)
        goodType &= seqLen->dataType() == DataType::INT32;
    if(hasInitH)
        goodType &= hI->dataType() == xDataType;
    if(hasInitC)
        goodType &= cI->dataType() == xDataType;

    if(!goodType)
        return false;




}



}
}
}
