op_library(tensorrt_engine_op DEPS tensorrt_engine tensorrt_converter
           infer_io_utils analysis_helper)
file(APPEND ${pybind_file} "USE_NO_KERNEL_OP__(tensorrt_engine);\n")
file(APPEND ${pybind_file} "USE_OP__(load_combine);\n")
file(APPEND ${pybind_file} "USE_OP__(load);\n")
#file(APPEND ${pybind_file} "USE_OP_ITSELF__(squeeze2);\n")
file(APPEND ${pybind_file} "USE_CUDA_ONLY_OP__(squeeze);\n")
#file(APPEND ${pybind_file} "USE_OP__(elementwise_div);\n")
#file(APPEND ${pybind_file} "USE_CUDA_ONLY_OP__(slice);\n")
file(APPEND ${pybind_file} "USE_CUDA_ONLY_OP__(expand);\n")
file(APPEND ${pybind_file} "USE_CPU_ONLY_OP__(multiclass_nms);\n")
file(APPEND ${pybind_file} "USE_CPU_ONLY_OP__(multiclass_nms2);\n")
file(APPEND ${pybind_file} "USE_CPU_ONLY_OP__(matrix_nms);\n")
file(APPEND ${pybind_file} "USE_CUDA_ONLY_OP__(box_coder);\n")
file(APPEND ${pybind_file} "USE_CUDA_ONLY_OP__(skip_layernorm);\n")
#file(APPEND ${pybind_file} "USE_CUDA_ONLY_OP__(reshape2);\n")
# new head
file(APPEND ${pybind_file} "USE_CUDA_ONLY_OP__(matmul);\n")
# plate_rec
file(APPEND ${pybind_file} "USE_CUDA_ONLY_OP__(im2sequence);\n")
file(APPEND ${pybind_file} "USE_CUDA_ONLY_OP__(gru);\n")
file(APPEND ${pybind_file} "USE_CUDA_ONLY_OP__(top_k);\n")
# library
file(APPEND ${pybind_file} "USE_CUDA_ONLY_OP__(density_prior_box);\n")
file(APPEND ${pybind_file}
     "USE_CUDA_ONLY_OP__(fusion_transpose_flatten_concat);\n")
file(APPEND ${pybind_file} "USE_CUDA_ONLY_OP__(flatten2);\n")
file(APPEND ${pybind_file} "USE_CUDA_ONLY_OP__(fc);\n")

#below are phi kernels
file(APPEND ${pybind_file} "USE_OP_ITSELF__(cast);\n")
file(APPEND ${pybind_file} "USE_OP_ITSELF__(shape);\n")
file(APPEND ${pybind_file} "USE_OP_ITSELF__(assign_value);\n")
file(APPEND ${pybind_file} "USE_OP_ITSELF__(fill_constant);\n")
file(APPEND ${pybind_file} "USE_OP_ITSELF__(transpose2);\n")
file(APPEND ${pybind_file} "USE_OP_ITSELF__(concat);\n")
file(APPEND ${pybind_file} "USE_OP_ITSELF__(split);\n")
file(APPEND ${pybind_file} "USE_OP_ITSELF__(scale);\n")
file(APPEND ${pybind_file} "USE_OP_ITSELF__(roi_align);\n")
file(APPEND ${pybind_file} "USE_OP_ITSELF__(conv2d);\n")
file(APPEND ${pybind_file} "USE_OP_ITSELF__(layer_norm);\n")
#file(APPEND ${pybind_file} "USE_CUDA_ONLY_OP__(layer_norm);\n")
# posegcn
# posegcn
file(APPEND ${pybind_file} "USE_OP_ITSELF__(elementwise_mul);\n")
# new plate
file(APPEND ${pybind_file} "USE_OP_ITSELF__(deformable_conv);\n")
file(APPEND ${pybind_file} "USE_OP_ITSELF__(pow);\n")
file(APPEND ${pybind_file} "USE_OP_ITSELF__(clip);\n")
file(APPEND ${pybind_file} "USE_OP_ITSELF__(log);\n")
# resnet50
file(APPEND ${pybind_file} "USE_OP_ITSELF__(softmax);\n")
# human_dec
file(APPEND ${pybind_file} "USE_OP_ITSELF__(yolo_box);\n")
file(APPEND ${pybind_file} "USE_OP_ITSELF__(batch_norm);\n")
file(APPEND ${pybind_file} "USE_OP_ITSELF__(elementwise_div);\n")
file(APPEND ${pybind_file} "USE_OP_ITSELF__(elementwise_add);\n")
# file(APPEND ${pybind_file} "USE_OP_ITSELF__(nearest_interp_v2);\n")
# face_reco
file(APPEND ${pybind_file} "USE_OP_ITSELF__(mul);\n")
# all object detect
file(APPEND ${pybind_file} "USE_OP_ITSELF__(gather);\n")
file(APPEND ${pybind_file} "USE_OP_ITSELF__(norm);\n")
file(APPEND ${pybind_file} "USE_OP_ITSELF__(reshape);\n")
file(APPEND ${pybind_file} "USE_OP_ITSELF__(sum);\n")

#fluid cuda kernel
file(APPEND ${pybind_file} "USE_CUDA_ONLY_OP__(nearest_interp);\n")

# 2023/3/30
file(APPEND ${pybind_file} "USE_OP_ITSELF__(top_k_v2);\n")
file(APPEND ${pybind_file} "USE_OP_ITSELF__(slice);;\n")
file(APPEND ${pybind_file} "USE_OP_ITSELF__(not_equal);;\n")
file(APPEND ${pybind_file} "USE_OP_ITSELF__(where_index);;\n")
file(APPEND ${pybind_file} "USE_OP_ITSELF__(gather_nd);;\n")

nv_test(
  test_tensorrt_engine_op
  SRCS tensorrt_engine_op_test.cc
  DEPS tensorrt_engine_op analysis)
