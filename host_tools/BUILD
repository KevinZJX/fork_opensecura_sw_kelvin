# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
