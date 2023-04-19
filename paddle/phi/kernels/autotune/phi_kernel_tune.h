// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include "paddle/fluid/platform/device_context.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_context.h"
#include "paddle/phi/core/kernel_factory.h"
#include "paddle/phi/kernels/autotune/gpu_timer.h"

namespace paddle {
namespace framework {
class OperatorBase;
}  // namespace framework
}  // namespace paddle
class PhiKernelTuner {
 public:
  explicit PhiKernelTuner(phi::KernelContext* ctx) : ctx_(ctx) {}
  virtual ~PhiKernelTuner() {}

  void AddPhiKernel(std::unique_ptr<phi::Kernel>&& kernel) {
    kernels_.push_back(std::forward<std::unique_ptr<phi::Kernel>>(kernel));
  }

  std::unique_ptr<phi::Kernel> Run() {
    PADDLE_ENFORCE_GT(
        kernels_.size(),
        0,
        phi::errors::InvalidArgument(
            "kernel num must be greater than 0, now is %d", kernels_.size()));
    std::lock_guard<std::mutex> lock(mutex_);
    size_t best_idx = 0;
    float min_time = std::numeric_limits<float>::max();

    // Time cost test estabulished in default stream.
    for (size_t i = 0; i < kernels_.size(); ++i) {
      auto time = RunAndMeasureKernel(kernels_[i].get(), ctx_);
      LOG(INFO) << "JZZ Run PhiKernelTune " << i << " time: " << time;
      if (time < min_time) {
        min_time = time;
        best_idx = i;
      }
    }
    LOG(INFO) << "JZZ Run best_idx: " << best_idx;
    return std::move(kernels_[best_idx]);
  }

 private:
  std::vector<std::unique_ptr<phi::Kernel>> kernels_;
  phi::KernelContext* ctx_;
  mutable std::mutex mutex_;

  float RunAndMeasureKernel(phi::Kernel* kernel, phi::KernelContext* ctx) {
    // Regard 1st run as warmup, judge the compare result by the time cost
    // of rest cycles.
    constexpr int repeats = 6;
    phi::GpuTimer timer;
    float time_cost = 0;

    paddle::platform::DeviceContextPool& pool =
        paddle::platform::DeviceContextPool::Instance();
    paddle::platform::CUDAPlace place(paddle::platform::GetCurrentDeviceId());
    auto* dev_ctx = static_cast<phi::GPUContext*>(pool.Get(place));

    const auto& stream = dev_ctx->stream();

    dev_ctx->Wait();
    for (int i = 0; i < repeats; ++i) {
      timer.Start(stream);
      (*kernel)(ctx);
      timer.Stop(stream);
      auto time = timer.ElapsedTime();
      if (i > 0) {
        time_cost += time;
      }
    }
    return time_cost;
  }
};

// class OpsTuner {
//  public:
//   explicit OpsTuner(paddle::framework::Scope* scope) : scope_(scope) {}
//   virtual ~OpsTuner() {}

//   void AddPhiKernel(std::unique_ptr<phi::Kernel>&& kernel) {
//     kernels_.push_back(std::forward<std::unique_ptr<phi::Kernel>>(kernel));
//   }

//   std::string Run() {
//     PADDLE_ENFORCE_GT(
//         ops_.size(),
//         0,
//         phi::errors::InvalidArgument(
//             "Operator num must be greater than 0, now is %d", ops_.size()));

//     std::lock_guard<std::mutex> lock(mutex_);
//     size_t best_idx = 0;
//     float min_time = std::numeric_limits<float>::max();

//     // Time cost test estabulished in default stream.
//     for (size_t i = 0; i < ops_.size(); ++i) {
//       auto time = RunAndMeasureKernel(kernels_[i].get(), ctx_);
//       if (time < min_time) {
//         min_time = time;
//         best_idx = i;
//       }
//     }
//     return std::move(kernels_[best_idx]);
//   }

//  private:
//   std::vector<std::pair<std::string, std::vector<std::unique_ptr<OperatorBase>>>> ops_;
//   paddle::framework::Scope* scope_;
//   mutable std::mutex mutex_;

//   float RunAndMeasureOps(phi::Kernel* kernel, phi::KernelContext* ctx) {
//     // Regard 1st run as warmup, judge the compare result by the time cost
//     // of rest cycles.
//     constexpr int repeats = 6;
//     phi::GpuTimer timer;
//     float time_cost = 0;

//     paddle::platform::DeviceContextPool& pool =
//         paddle::platform::DeviceContextPool::Instance();
//     paddle::platform::CUDAPlace place(paddle::platform::GetCurrentDeviceId());
//     auto* dev_ctx = static_cast<phi::GPUContext*>(pool.Get(place));

//     const auto& stream = dev_ctx->stream();

//     dev_ctx->Wait();
//     for (int i = 0; i < repeats; ++i) {
//       timer.Start(stream);
//       (*kernel)(ctx);
//       timer.Stop(stream);
//       auto time = timer.ElapsedTime();
//       if (i > 0) {
//         time_cost += time;
//       }
//     }
//     return time_cost;
//   }
// };