def _openvino_native_impl(repository_ctx):
    openvino_native_dir = repository_ctx.os.environ.get("OPENVINO_NATIVE_DIR")
    repository_ctx.symlink(openvino_native_dir, "openvino")
    repository_ctx.file("BUILD", """
exports_files(["openvino"])
cc_library(
    name = "openvino",
    hdrs = glob(["${sysroot}/usr/local/include/ie",
    "${sysroot}/usr/local/include/ie/cpp",
    "${sysroot}/usr/local/include/openvino",
    "${sysroot}/usr/local/runtime/include/ie",
    "${sysroot}/usr/local/runtime/include/ie/cpp",
    "${sysroot}/usr/local/runtime/include/"
    ]),
    srcs = glob([
        "libopenvino.so",
    ]),
    visibility = ["//visibility:public"],
)
    """)

openvino_configure = repository_rule(
    implementation = _openvino_native_impl,
    local = True,
    environ = [
        "OPENVINO_NATIVE_DIR"
    ])
