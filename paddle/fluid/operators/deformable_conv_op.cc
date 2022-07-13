// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <memory>
#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/multiary.h"

namespace paddle {
namespace operators {
class DeformableConvOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Input",
             "(Tensor) The input of deformable conv op. "
             "The shape of input is "
             "[N, channel_in, H, W]");
    AddInput("Offset",
             "(Tensor) The input offset. "
             "The shape of the offset is "
             "[N, deformable_groups * kernel_w * kernel_h * 2, H, W");
    AddInput("Mask",
             "(Tensor) The input mask. "
             "The shape of the mask is "
             "[N, deformable_groups * kernel_w * kernel_h, H, W].");
    AddInput("Filter",
             "(Tensor) The Input Filter "
             "The shape of the wight is "
             "[num_filters, channel_in, kernel_h, kernel_w.");
    AddOutput("Output",
              "(Tensor) The output. "
              "The shape of the output tensor is "
              "[N, num_filters, out_height, out_width]].");
    AddAttr<std::vector<int>>("strides",
                              "(vector<int> default:{1, 1}), the "
                              "strides(h_stride, w_stride) of "
                              "convolution operator.")
        .SetDefault({1, 1});
    AddAttr<std::vector<int>>("paddings",
                              "(vector<int> default:{0,0}), the "
                              "paddings(h_pad, w_pad) of "
                              "convolution operator. ")
        .SetDefault({0, 0});
    AddAttr<std::vector<int>>("dilations",
                              "(vector<int> default:{1, 1}), the "
                              "dilations(h_dilation, w_dilation) of "
                              "convolution operator.")
        .SetDefault({1, 1});
    AddAttr<int>(
        "groups",
        "(int default:1), the groups number of the convolution operator. "
        "According to grouped convolution in Alex Krizhevsky's Deep CNN paper: "
        "when group=2, the first half of the filters is only connected to the "
        "first half of the input channels, while the second half of the "
        "filters "
        "is only connected to the second half of the input channels.")
        .SetDefault(1);
    AddAttr<int>("deformable_groups",
                 "(int default:1), the number of the deformable groups.")
        .SetDefault(1);
    AddAttr<int>("im2col_step",
                 "im2col maximum number of image per computation")
        .SetDefault(64);
    AddComment(R"DOC(
**Deformable Convolution Operator**

Compute 2-D deformable convolution on 4-D input.

Given input image x, output feature map y, the deformable convolution operation can be expressed as follow:

$$
y(p) = \\sum_{k=1}^{K}{w_k * x(p + p_k + \\Delta p_k) * \\Delta m_k}
$$

Where $$\\Delta p_k$$ and $$\Delta m_k$$ are the learnable offset and modulation scalar for the k-th location, respectively.

Refer to 'Deformable ConvNets v2: More Deformable, Better Results
'<https://arxiv.org/abs/1811.11168v2>

Example:
  Input:
       Input shape: $(N, C_{in}, H_{in}, W_{in})$
       Filter shape: $(C_{out}, C_{in}, H_f, W_f)$
       Offset shape: $(N, 2 * deformable_groups, * H_f * W_f, H_{out}, W_{out})$
       Mask shape: $(N, deformable_groups * H_f * W_f, H_{out}, W_{out})$
  Output:
       Output shape: $(N, C_{out}, H_{out}, W_{out})$
                     where $H_{out}, W_{out}$ must be equal to $H_{in}, W_{in}$ respectively.
  Where
$$
       H_{out}= \frac{(H_{in} + 2 * paddings[0] - (dilations[0] * (H_f - 1) + 1))}{strides[0]}+ 1 \\
       W_{out}= \frac{(W_{in} + 2 * paddings[1] - (dilations[1] * (W_f - 1) + 1))}{strides[1]}+ 1
$$
)DOC");
  }
};

class DeformableConvOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "Input"),
        ctx.device_context());
  }
};

template <typename T>
class DeformableConvGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("deformable_conv_grad");
    op->SetInput("Input", this->Input("Input"));
    op->SetInput("Filter", this->Input("Filter"));
    op->SetInput("Offset", this->Input("Offset"));
    op->SetInput("Mask", this->Input("Mask"));
    op->SetInput(framework::GradVarName("Output"), this->OutputGrad("Output"));

    op->SetOutput(framework::GradVarName("Input"), this->InputGrad("Input"));
    op->SetOutput(framework::GradVarName("Filter"), this->InputGrad("Filter"));
    op->SetOutput(framework::GradVarName("Offset"), this->InputGrad("Offset"));
    op->SetOutput(framework::GradVarName("Mask"), this->InputGrad("Mask"));

    op->SetAttrMap(this->Attrs());
  }
};

class DeformableConvGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    auto in_dims = ctx->GetInputDim("Input");
    auto filter_dims = ctx->GetInputDim("Filter");
    auto offset_dims = ctx->GetInputDim("Offset");
    auto mask_dims = ctx->GetInputDim("Mask");

    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Output")), "Input",
                   "Output@Grad", "deformable_conv_grad");
    if (ctx->HasOutput(framework::GradVarName("Input"))) {
      ctx->SetOutputDim(framework::GradVarName("Input"), in_dims);
    }
    if (ctx->HasOutput(framework::GradVarName("Filter"))) {
      ctx->SetOutputDim(framework::GradVarName("Filter"), filter_dims);
    }
    if (ctx->HasOutput(framework::GradVarName("Offset"))) {
      ctx->SetOutputDim(framework::GradVarName("Offset"), offset_dims);
    }
    if (ctx->HasOutput(framework::GradVarName("Mask"))) {
      ctx->SetOutputDim(framework::GradVarName("Mask"), mask_dims);
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "Input"),
        ctx.device_context());
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
DECLARE_INFER_SHAPE_FUNCTOR(deformable_conv, DeformableConvInferShapeFunctor,
                            PD_INFER_META(phi::DeformableConvInferMeta));

REGISTER_OPERATOR__(deformable_conv, ops::DeformableConvOp,
                  ops::DeformableConvOpMaker,
                  ops::DeformableConvGradOpMaker<paddle::framework::OpDesc>,
                  ops::DeformableConvGradOpMaker<paddle::imperative::OpBase>,
                  DeformableConvInferShapeFunctor);

REGISTER_OPERATOR(deformable_conv_grad, ops::DeformableConvGradOp);
