package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "operation_builder",
    srcs = [
        "add.cc",
	"conv2d.cc",
    ],
    hdrs = [
        "add.h",
	"conv2d.h",
	"operations_base.h",
	"openvino_node_manager.h",
    ],
    tags = [
        "manual",
        "nobuilder",
    ],
    deps = [
        "//tensorflow/lite:kernel_api",
        "//tensorflow/lite:util",
        "//tensorflow/lite/c:common",
	"//tensorflow/lite/tools:logging",
        "//tensorflow/lite/kernels:kernel_util",
        "//tensorflow/lite/kernels:padding",
        "//tensorflow/lite/kernels/internal:optimized_base",
        "//tensorflow/lite/kernels/internal:tensor",
        "//tensorflow/lite/kernels/internal:types",
        "@intel_openvino//:openvino",
    ],
)

