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
#include<ops/declarable/helpers/transforms.h>

namespace nd4j      {
namespace ops       {
namespace platforms {

//////////////////////////////////////////////////////////////////////////
static void lstmLayerCUDNN(const LaunchContext* context,
                        const NDArray* x, const NDArray* w, const NDArray* b, const NDArray* seqLen, const NDArray* hI, const NDArray* cI,
                        NDArray* h, NDArray* hL, NDArray* cL,
                        const std::vector<float>& params) {
    // notations:
    // bS - batch size
    // sL - sequence length, number of time steps
    // nIn - input size
    // nOut - output size (hidden size)

    //     INPUTS:

    // *******
    // input x:
    // 1) [bS, nIn, sL]  when dataFormat == 2

    // *******
    // weights (input + recurrent) w:
    // 1) [nIn + nOut, 4*nOut]    when directionMode <  2
    // 2) [2*(nIn + nOut), 4*nOut] when directionMode >= 2

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
    // 1) [bS, nOut, sL]    when directionMode <= 2 && dataFormat == 2
    // 2) [bS, 2*nOut, sL]  when directionMode == 3 && dataFormat == 2

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

    auto handle = reinterpret_cast<cudnnHandle_t *>(context->getCuDnnHandle());
    cudnnStatus_t err = cudnnSetStream(*handle, *context->getCudaStream());
    if (err != 0) throw nd4j::cuda_exception::build("conv2dCUDNN: can't set stream for cuDNN", err);

    const int numDims = 3;

    const int numOfDirections = (params[1] /*directionMode*/ == 0 ) ? 1 : 2;

    const int bS   = x->sizeAt(0);
    const int nIn  = x->sizeAt(1);
    const int sL   = x->sizeAt(2);
    const int nOut = w->sizeAt(-1) / 4;

    int* seqLengthArray = nullptr;
    if(seqLen != nullptr)
        seqLengthArray = seqLen->bufferAsT<int>();
    else {
        seqLengthArray = new int[bS];
        for (uint i = 0; i < bS; ++i)
            seqLengthArray[i] = sL;
    }

    // const std::vector<int> xShape = {bS, nIn, sL};
    // const std::vector<int> hShape = {bS, numOfDirections*nOut, sL};
    const std::vector<int> wShape = {1, numOfDirections*(nIn + nOut), 4*nOut};
    const std::vector<int> hIcIShape = {numOfDirections, bS, nOut};

    cudnnTensorFormat_t format = CUDNN_TENSOR_NCHW;

    // x descriptor
    cudnnRNNDataDescriptor_t xDesc;
    cudnnCreateRNNDataDescriptor(&xDesc);
    err = cudnnSetRNNDataDescriptor(xDesc, cudnnDataType(x->dataType()), CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED, sL, bS, nIn, seqLengthArray, nullptr);
    if(err != 0) throw nd4j::cuda_exception::build("lstmLayerCUDNN: cudnnSetRNNDataDescriptor for x failed", err);

    // hI/cI/hL/cL, descriptor is same for all of them
    cudnnTensorDescriptor_t hcDesc;
    if(hI != nullptr || cI != nullptr || cL != nullptr || cL != nullptr) {
        cudnnCreateTensorDescriptor(&hcDesc);
        err = cudnnSetTensorNdDescriptorEx(hcDesc, format, cudnnDataType(hI != nullptr ? hI->dataType() : cI->dataType()), numDims, hIcIShape.data());
        if(err != 0) throw nd4j::cuda_exception::build("lstmLayerCUDNN: cudnnSetFilterNdDescriptor for hI/cI/hL/cL failed", err);
    }

    // weights descriptor
    cudnnFilterDescriptor_t wDesc;
    cudnnCreateFilterDescriptor(&wDesc);
    err = cudnnSetFilterNdDescriptor(wDesc, cudnnDataType(w->dataType()), format, numDims, wShape.data());
    if(err != 0) throw nd4j::cuda_exception::build("lstmLayerCUDNN: cudnnSetFilterNdDescriptor for weights failed", err);

    // h descriptor
    cudnnRNNDataDescriptor_t hDesc;
    cudnnCreateRNNDataDescriptor(&hDesc);
    err = cudnnSetRNNDataDescriptor(hDesc, cudnnDataType(h->dataType()), CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED, sL, bS, nOut, seqLengthArray, nullptr);
    if(err != 0) throw nd4j::cuda_exception::build("lstmLayerCUDNN: cudnnSetRNNDataDescriptor for h failed", err);

    // no dropout descriptor
    cudnnDropoutDescriptor_t dropoutDesc;

    // description of lstm
    cudnnDataType_t typeOfData = CUDNN_DATA_FLOAT;
    if(x->dataType() == DataType::DOUBLE)
        typeOfData = CUDNN_DATA_DOUBLE;
    else if(x->dataType() == DataType::HALF)
        typeOfData = CUDNN_DATA_HALF;

    cudnnRNNDescriptor_t lstmDesc;
    cudnnCreateRNNDescriptor(&lstmDesc);
    err = cudnnSetRNNDescriptor(*handle, lstmDesc, nOut, 1, dropoutDesc, CUDNN_LINEAR_INPUT,
                                (numOfDirections == 1) ? CUDNN_UNIDIRECTIONAL : CUDNN_BIDIRECTIONAL,
                                CUDNN_LSTM, CUDNN_RNN_ALGO_STANDARD, typeOfData);
    if (err != 0) throw nd4j::cuda_exception::build("lstmLayerCUDNN: cudnnSetRNNDescriptor failed", err);

    // set clip value
    if(params[2] != 0) {    // params[2] == clipValue
        err = cudnnRNNSetClip(*handle, lstmDesc, CUDNN_RNN_CLIP_MINMAX, CUDNN_NOT_PROPAGATE_NAN, -params[2], params[2]);
        if (err != 0) throw nd4j::cuda_exception::build("lstmLayerCUDNN: cudnnRNNSetClip failed", err);
    }

    // set bias mode
    // if(b != nullptr) {
    //     err = cudnnStatus_t cudnnSetRNNBiasMode(lstmDesc, CUDNN_RNN_SINGLE_INP_BIAS);
    //     if (err != 0) throw nd4j::cuda_exception::build("lstmLayerCUDNN: cudnnSetRNNBiasMode failed", err);
    // }

    // allocate amount of device memory necessary for lstm calculation process
    size_t workSpaceSizeInBytes;
    workSpaceSizeInBytes = x->lengthOf() * x->sizeOfT();
    // err = cudnnGetRNNWorkspaceSize(*handle, lstmDesc, sL, xDesc, &workSpaceSizeInBytes);
    // if (err != 0) throw nd4j::cuda_exception::build("lstmLayerCUDNN: cudnnGetRNNWorkspaceSize failed", err);
    void* workSpace;
    auto cudaErr = cudaMalloc(&workSpace, workSpaceSizeInBytes);
    if (cudaErr != 0) throw nd4j::cuda_exception::build("lstmLayerCUDNN: cudaMalloc for auxiliary workspace memory failed", cudaErr);


    NDArray::prepareSpecialUse({h, hL, cL}, {x, w, b, seqLen, hI, cI});

    // run calculation
    err = cudnnRNNForwardInferenceEx(*handle, lstmDesc,
                                     xDesc, x->getSpecialBuffer(),
                                     hcDesc, hI ? hI->getSpecialBuffer() : nullptr,
                                     hcDesc, cI ? cI->getSpecialBuffer() : nullptr,
                                     wDesc, w->getSpecialBuffer(),
                                     hDesc, h->getSpecialBuffer(),
                                     hcDesc, hL ? hL->getSpecialBuffer() : nullptr,
                                     hcDesc, cL ? cL->getSpecialBuffer() : nullptr,
                                     nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                     workSpace, workSpaceSizeInBytes);
    if (err != 0) throw nd4j::cuda_exception::build("lstmLayerCUDNN: cudnnRNNForwardInferenceEx failed", err);

    cudaErr = cudaStreamSynchronize(*context->getCudaStream());
    if (cudaErr != 0)
        throw cuda_exception::build("lstmLayerCUDNN: cudaStreamSynchronize failed !", cudaErr);

    NDArray::registerSpecialUse({h, hL, cL}, {x, w, b, seqLen, hI, cI});

    if(seqLen = nullptr)
        delete []seqLengthArray;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(lstmLayer, ENGINE_CUDA) {

   // equations (no peephole connections)
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

    const auto dataFormat    = INT_ARG(0);    // for unidirectional: 0 = [sL, bS, nIn], 1 = [bS, sL ,nIn], 2 = [bS, nIn, sL], for bidirectional: 3 = [sL, 2, bS, nOut] (ONNX)
    const auto directionMode = INT_ARG(1);    // direction: 0 = fwd, 1 = bwd, 2 = bidirectional sum, 3 = bidirectional concat, 4 = bidirectional extra output dim (in conjunction with format dataFormat = 3)

    // integer numbers corresponding to activations: 0=tanh, 1=relu, 2=sigmoid, 3=affine, 4=leaky relu, 5= thresholded relu, 6=scaled tanh, 7=hard sigmoid, 8=ELU, 9=softsign, 10=softplus
    const auto gateAct       = INT_ARG(2);    // activation for input (i), forget (f) and output (o) gates
    const auto cellAct       = INT_ARG(3);    // activation for cell state (c')
    const auto outAct        = INT_ARG(4);    // activation for output (h)

    const auto hasBiases  = B_ARG(0);   // indicates whether biases array is provided
    const auto hasSeqLen  = B_ARG(1);   // indicates whether seqLen array is provided
    const auto hasInitH   = B_ARG(2);   // indicates whether initial output is provided
    const auto hasInitC   = B_ARG(3);   // indicates whether initial cell state is provided
    const auto hasPH      = B_ARG(4);   // indicates whether peephole connections are present
    const auto retFullSeq = B_ARG(5);   // indicates whether to return whole time sequence h {h_0, h_1, ... , h_sL-1}
    const auto retLastH   = B_ARG(6);   // indicates whether to return output at last time step only, in this case shape would be [bS, nOut] (exact shape depends on dataFormat argument)
    const auto retLastC   = B_ARG(7);   // indicates whether to return cells state at last time step only, in this case shape would be [bS, nOut] (exact shape depends on dataFormat argument)

    const auto gateActHasAlpha = gateAct == 3 || gateAct == 4 || gateAct == 5 || gateAct == 6 || gateAct == 8;
    const auto cellActHasAlpha = cellAct == 3 || cellAct == 4 || cellAct == 5 || cellAct == 6 || cellAct == 8;
    const auto outActHasAlpha  = outAct  == 3 || outAct  == 4 || outAct  == 5 || outAct  == 6 || outAct  == 8;
    const auto gateActHasBeta  = gateAct == 3 || gateAct == 6;
    const auto cellActHasBeta  = cellAct == 3 || cellAct == 6;
    const auto outActHasBeta   = outAct  == 3 || outAct  == 6;

    uint count = 1;
    const auto cellClip = T_ARG(0);                                     // cell clipping value, if it = 0 then do not apply clipping
    const auto gateAlpha = gateActHasAlpha ? T_ARG(count++) : 0;
    const auto gateBeta  = gateActHasBeta  ? T_ARG(count++) : 0;
    const auto cellAlpha = cellActHasAlpha ? T_ARG(count++) : 0;
    const auto cellBeta  = cellActHasBeta  ? T_ARG(count++) : 0;
    const auto outAlpha  = outActHasAlpha  ? T_ARG(count++) : 0;
    const auto outBeta   = outActHasBeta   ? T_ARG(count++) : 0;

    const auto x  = INPUT_VARIABLE(0);          // input
    const auto Wx = INPUT_VARIABLE(1);          // input weights
    const auto Wr = INPUT_VARIABLE(2);          // recurrent weights

    count = 3;
    const auto b      = hasBiases ? INPUT_VARIABLE(count++) : nullptr;  // biases
    const auto seqLen = hasSeqLen ? INPUT_VARIABLE(count++) : nullptr;  // seqLen vector
    const auto hI     = hasInitH  ? INPUT_VARIABLE(count++) : nullptr;  // initial output
    const auto cI     = hasInitC  ? INPUT_VARIABLE(count++) : nullptr;  // initial cell state
    const auto Wp     = hasPH     ? INPUT_VARIABLE(count++) : nullptr;  // peephole weights

    REQUIRE_TRUE(dataFormat < 3 || (dataFormat == 3 && directionMode == 4), 0, "LSTM_LAYER CUDNN operation: if argument dataFormat = 3, then directionMode = 4, but got dataFormat = %i and directionMode = %i instead !", dataFormat, directionMode);
    REQUIRE_TRUE(cellClip >= 0 , 0, "LSTM_LAYER CUDNN operation: cell clipping value should be nonnegative (>=0) !");
    REQUIRE_TRUE(retFullSeq || retLastH || retLastC, 0, "LSTM_LAYER CUDNN operation: please specify what output arrays to produce !");

    count = 0;
    auto h  = retFullSeq ? OUTPUT_VARIABLE(count++) : nullptr;           // output
    auto hL = retLastH   ? OUTPUT_VARIABLE(count++) : nullptr;           // output at last step
    auto cL = retLastC   ? OUTPUT_VARIABLE(count++) : nullptr;           // cell state at last step

    // evaluate dimensions
    const Nd4jLong sL   = dataFormat == 3 ?  x->sizeAt(0) : x->sizeAt(dataFormat);
    const Nd4jLong bS   = dataFormat == 1 || dataFormat == 2 ? x->sizeAt(0) : x->sizeAt(-2);
    const Nd4jLong nIn  = dataFormat == 2 ? x->sizeAt(1) : x->sizeAt(-1);
    const Nd4jLong nOut = Wx->sizeAt(-1) / 4;

    // inputs validations
    if(directionMode < 2) {     // no bidirectional

        // Wx validation
        if(Wx->rankOf() != 2 || Wx->sizeAt(0) != nIn)
            REQUIRE_TRUE(false, 0, "LSTM_LAYER CUDNN operation: wrong shape of input weights, expected is %s, but got %s instead !", ShapeUtils::shapeAsString({nIn, 4*nOut}).c_str(), ShapeUtils::shapeAsString(Wx).c_str());
        // Wr validation
        if(Wr->rankOf() != 2 || Wr->sizeAt(0) != nOut || Wr->sizeAt(1) != 4*nOut)
            REQUIRE_TRUE(false, 0, "LSTM_LAYER CUDNN operation: wrong shape of recurrent weights, expected is %s, but got %s instead !", ShapeUtils::shapeAsString({nOut, 4*nOut}).c_str(), ShapeUtils::shapeAsString(Wr).c_str());
        // biases validation
        if(b != nullptr && (b->rankOf() != 1 || b->sizeAt(0) != 4*nOut))
            REQUIRE_TRUE(false, 0, "LSTM_LAYER CUDNN operation: wrong shape of biases, expected is %s, but got %s instead !", ShapeUtils::shapeAsString({4*nOut}).c_str(), ShapeUtils::shapeAsString(b).c_str());
        // initial output validation
        if(hI != nullptr && (hI->rankOf() != 2 || hI->sizeAt(0) != bS || hI->sizeAt(1) != nOut))
            REQUIRE_TRUE(false, 0, "LSTM_LAYER CUDNN operation: wrong shape of initial output, expected is %s, but got %s instead !", ShapeUtils::shapeAsString({bS, nOut}).c_str(), ShapeUtils::shapeAsString(hI).c_str());
        // initial cell  validation
        if(cI != nullptr && (cI->rankOf() != 2 || cI->sizeAt(0) != bS || cI->sizeAt(1) != nOut))
            REQUIRE_TRUE(false, 0, "LSTM_LAYER CUDNN operation: wrong shape of initial cell state, expected is %s, but got %s instead !", ShapeUtils::shapeAsString({bS, nOut}).c_str(), ShapeUtils::shapeAsString(cI).c_str());
    }
    else {                  // bidirectional
         // Wx validation
        if(Wx->rankOf() != 3 || Wx->sizeAt(0) != 2 || Wx->sizeAt(1) != nIn)
            REQUIRE_TRUE(false, 0, "LSTM_LAYER CUDNN operation: wrong shape of input weights, expected is %s, but got %s instead !", ShapeUtils::shapeAsString({2, nIn, 4*nOut}).c_str(), ShapeUtils::shapeAsString(Wx).c_str());
        // Wr validation
        if(Wr->rankOf() != 3 || Wr->sizeAt(0) != 2 || Wr->sizeAt(1) != nOut || Wr->sizeAt(2) != 4*nOut)
            REQUIRE_TRUE(false, 0, "LSTM_LAYER CUDNN operation: wrong shape of recurrent weights, expected is %s, but got %s instead !", ShapeUtils::shapeAsString({2, nOut, 4*nOut}).c_str(), ShapeUtils::shapeAsString(Wr).c_str());
        // biases validation
        if(b != nullptr && (b->rankOf() != 2 || b->sizeAt(0) != 2 || b->sizeAt(1) != 4*nOut))
            REQUIRE_TRUE(false, 0, "LSTM_LAYER CUDNN operation: wrong shape of biases, expected is %s, but got %s instead !", ShapeUtils::shapeAsString({2, 4*nOut}).c_str(), ShapeUtils::shapeAsString(b).c_str());
        // initial output validation
        if(hI != nullptr && (hI->rankOf() != 3 || hI->sizeAt(0) != 2 || hI->sizeAt(1) != bS || hI->sizeAt(2) != nOut))
            REQUIRE_TRUE(false, 0, "LSTM_LAYER CUDNN operation: wrong shape of initial output, expected is %s, but got %s instead !", ShapeUtils::shapeAsString({2, bS, nOut}).c_str(), ShapeUtils::shapeAsString(hI).c_str());
        // initial cell  validation
        if(cI != nullptr && (cI->rankOf() != 3 || cI->sizeAt(0) != 2 || cI->sizeAt(1) != bS || cI->sizeAt(2) != nOut))
            REQUIRE_TRUE(false, 0, "LSTM_LAYER CUDNN operation: wrong shape of initial cell state, expected is %s, but got %s instead !", ShapeUtils::shapeAsString({2, bS, nOut}).c_str(), ShapeUtils::shapeAsString(cI).c_str());
    }

    std::vector<float> params = {static_cast<float>(dataFormat), static_cast<float>(directionMode), static_cast<float>(cellClip),
                                 static_cast<float>(gateAct), static_cast<float>(gateAlpha), static_cast<float>(gateBeta),
                                 static_cast<float>(cellAct), static_cast<float>(cellAlpha), static_cast<float>(cellBeta),
                                 static_cast<float>(outAct), static_cast<float>(outAlpha), static_cast<float>(outBeta)};

    const uint numOfDirections = (directionMode < 2) ? 1 : 2;

    // cudnn requires only one weights array
    NDArray w(Wx->ordering(), {numOfDirections * (nIn + nOut), 4*nOut}, Wx->dataType(), Wx->getContext());
    if(numOfDirections == 1)
        helpers::concat(block.launchContext(), {Wx, Wr}, w, 0);
    else {
        NDArray WxForward  = (*Wx)({0,1, 0,0, 0,0});
        NDArray WrForward  = (*Wr)({0,1, 0,0, 0,0});
        NDArray WxBackward = (*Wx)({1,2, 0,0, 0,0});
        NDArray WrBackward = (*Wr)({1,2, 0,0, 0,0});

        helpers::concat(block.launchContext(), {&WxForward, &WrForward, &WxBackward, &WrBackward}, w, 0);  // nIn + nOut + nIn + nOut = 2 * (nIn + nOut)
        // w({0,nIn,    0,0}).assign((*Wx)({0,1, 0,0, 0,0}));
        // w({nIn,nOut, 0,0}).assign((*Wr)({0,1, 0,0, 0,0}));
        // w({nIn+nOut,2*nIn+nOut, 0,0}).assign((*Wx)({1,2, 0,0, 0,0}));
        // w({2*nIn+nOut,-1,       0,0}).assign((*Wr)({1,2, 0,0, 0,0}));
    }

    lstmLayerCUDNN(block.launchContext(), x, &w, b, seqLen, hI, cI, h, hL, cL, params);

    return Status::OK();
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(lstmLayer, ENGINE_CUDA) {

    const auto x = INPUT_VARIABLE(0);
    const auto Wx = INPUT_VARIABLE(1);
    const auto Wr = INPUT_VARIABLE(2);

    const auto dataFormat    = INT_ARG(0);    // for unidirectional: 0 = [sL, bS, nIn], 1 = [bS, sL ,nIn], 2 = [bS, nIn, sL], for bidirectional: 3 = [sL, 2, bS, nOut] (ONNX)
    const auto directionMode = INT_ARG(1);    // direction: 0 = fwd, 1 = bwd, 2 = bidirectional sum, 3 = bidirectional concat, 4 = bidirectional extra output dim (in conjunction with format dataFormat = 3)

    if(dataFormat != 2 || (directionMode != 0 && directionMode != 3))
        return false;

    const auto gateAct = INT_ARG(2);    // activation for input (i), forget (f) and output (o) gates, 2==sigmoid is supported only
    const auto cellAct = INT_ARG(3);    // activation for cell state (c'), 0==tanh is supported only
    const auto outAct  = INT_ARG(4);    // activation for output (h), 0==tanh is supported only

    if(gateAct != 2 || cellAct != 0 || outAct != 0)
        return false;

    const auto hasBiases  = B_ARG(0);   // indicates whether biases array is provided
    const auto hasSeqLen  = B_ARG(1);   // indicates whether seqLen array is provided
    const auto hasInitH   = B_ARG(2);   // indicates whether initial output is provided
    const auto hasInitC   = B_ARG(3);   // indicates whether initial cell state is provided
    const auto hasPH      = B_ARG(4);   // indicates whether peephole connections are present
    const auto retFullSeq = B_ARG(5);   // indicates whether to return whole time sequence h {h_0, h_1, ... , h_sL-1}
    const auto retLastH   = B_ARG(6);   // indicates whether to return output at last time step only, in this case shape would be [bS, nOut] (exact shape depends on dataFormat argument)
    const auto retLastC   = B_ARG(7);   // indicates whether to return cells state at last time step only, in this case shape would be [bS, nOut] (exact shape depends on dataFormat argument)

    if(hasPH || !retFullSeq)
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

    if(x->ews() != 1)
        return false;

    if(hasInitH && hI->ews() != 1)
        return false;

    if(hasInitC && cI->ews() != 1)
        return false;

    if(h->ews() != 1)
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

    return true;
}



}
}
}
