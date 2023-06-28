
cc_binary(
    name = "encoding_bin",
    srcs = ["encoding_main.cc"],
)

genrule(
    name = "encoding_files",
    srcs = [],
    outs = [
        "kelvin-opc.h",
        "kelvin-opc.c",
        "kelvin_intrinsics.h",
    ],
# encoding_bin is in bazel-out/host/ because of cross-compile. We need to move
# the outputs to $RULEDIR
    cmd = """
    mkdir -p $(RULEDIR)
    ./$(location :encoding_bin)
    mv kelvin-opc.c $(RULEDIR)
    mv kelvin-opc.h $(RULEDIR)
    mv kelvin_intrinsics.h $(RULEDIR)
    """,
    tools = [":encoding_bin"],
)
