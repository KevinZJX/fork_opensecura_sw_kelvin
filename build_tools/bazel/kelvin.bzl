"""Rules to build Kelvin SW objects"""

load("@rules_cc//cc:find_cc_toolchain.bzl", "find_cc_toolchain")

KELVIN_PLATFORM = "//platforms/riscv32:kelvin"

def _kelvin_transition_impl(_settings, _attr):
    return {"//command_line_option:platforms": KELVIN_PLATFORM}

kelvin_transition = transition(
    implementation = _kelvin_transition_impl,
    inputs = [],
    outputs = ["//command_line_option:platforms"],
)

def kelvin_rule(**kwargs):
    """Kelvin-specific transition rule.

    A wrapper over rule() for creating rules that trigger
    the transition to the kelvin platform config.

    Args:
      **kwargs: params forwarded to the implementation.
    Returns:
      Kelvin transition rule.
    """
    attrs = kwargs.pop("attrs", {})
    if "platform" not in attrs:
        attrs["platform"] = attr.string(default = KELVIN_PLATFORM)
    attrs["_allowlist_function_transition"] = attr.label(
        default = "@bazel_tools//tools/allowlists/function_transition_allowlist",
    )

    return rule(
        cfg = kelvin_transition,
        attrs = attrs,
        **kwargs
    )

def _kelvin_binary_impl(ctx):
    """Implements compilation for kelvin executables.

    This rule compiles and links provided input into an executable
    suitable for use on the Kelvin core. Generates both an ELF
    and a BIN.

    Args:
      ctx: context for the rules.
        srcs: Input source files.
        deps: Target libraries that the binary depends upon.
        hdrs: Header files that are local to the binary.
        copts: Flags to pass along to the compiler.
        defines: Preprocessor definitions.
        linkopts: Flags to pass along to the linker.

    Output:
        OutputGroupsInfo to allow definition of filegroups
        containing the output ELF and BIN.
    """
    cc_toolchain = find_cc_toolchain(ctx).cc
    feature_configuration = cc_common.configure_features(
        ctx = ctx,
        cc_toolchain = cc_toolchain,
        requested_features = ctx.features,
        unsupported_features = ctx.disabled_features,
    )
    compilation_contexts = []
    linking_contexts = []
    for dep in ctx.attr.deps:
        if CcInfo in dep:
            compilation_contexts.append(dep[CcInfo].compilation_context)
            linking_contexts.append(dep[CcInfo].linking_context)
    (_compilation_context, compilation_outputs) = cc_common.compile(
        actions = ctx.actions,
        cc_toolchain = cc_toolchain,
        feature_configuration = feature_configuration,
        name = ctx.label.name,
        srcs = ctx.files.srcs,
        compilation_contexts = compilation_contexts,
        private_hdrs = ctx.files.hdrs,
        user_compile_flags = ctx.attr.copts,
        defines = ctx.attr.defines,
    )
    linking_outputs = cc_common.link(
        name = "{}.elf".format(ctx.label.name),
        actions = ctx.actions,
        feature_configuration = feature_configuration,
        cc_toolchain = cc_toolchain,
        compilation_outputs = compilation_outputs,
        linking_contexts = linking_contexts,
        user_link_flags = ctx.attr.linkopts + [
            "-Wl,-T,{}".format(ctx.file.linker_script.path),
            "-Wl,--no-warn-rwx-segments",
        ],
        additional_inputs = depset([ctx.file.linker_script] + ctx.files.linker_script_includes),
        output_type = "executable",
    )

    binary = ctx.actions.declare_file(
        "{}.bin".format(
            ctx.attr.name,
        ),
    )
    ctx.actions.run(
        outputs = [binary],
        inputs = [linking_outputs.executable] + cc_toolchain.all_files.to_list(),
        arguments = [
            "-g",
            "-O",
            "binary",
            linking_outputs.executable.path,
            binary.path,
        ],
        executable = cc_toolchain.objcopy_executable,
    )

    return [
        DefaultInfo(
            files = depset([linking_outputs.executable, binary]),
        ),
        OutputGroupInfo(
            all_files = depset([linking_outputs.executable, binary]),
            elf_file = depset([linking_outputs.executable]),
            bin_file = depset([binary]),
        ),
    ]

kelvin_binary_impl = kelvin_rule(
    implementation = _kelvin_binary_impl,
    attrs = {
        "srcs": attr.label_list(allow_files = True),
        "deps": attr.label_list(allow_empty = True, providers = [CcInfo]),
        "hdrs": attr.label_list(allow_files = [".h"], allow_empty = True),
        "copts": attr.string_list(),
        "defines": attr.string_list(),
        "linkopts": attr.string_list(),
        "linker_script": attr.label(allow_single_file = True),
        "linker_script_includes": attr.label_list(default = [], allow_files = True),
        "_cc_toolchain": attr.label(default = Label("@bazel_tools//tools/cpp:current_cc_toolchain")),
    },
    fragments = ["cpp"],
    toolchains = ["@rules_cc//cc:toolchain_type"],
)

def kelvin_binary(
        name,
        srcs,
        is_riscv_test = False,
        **kwargs):
    """A helper macro for generating binary artifacts for the kelvin core.

    This macro uses the kelvin toolchain, kelvin-specific starting asm,
    and kelvin linker script to build kelvin binaries.

    Args:
      name: The name of this rule.
      srcs: The c source files.
      is_riscv_test: A bool flag to decide if a custom _start asm is included.
        It is used by riscv-tests.
      **kwargs: Additional arguments forward to cc_binary.
    Emits rules:
      filegroup              named: <name>.bin
        Containing the binary output for the target.
      filegroup              named: <name>.elf
        Containing all elf output for the target.
    """
    srcs.append("//crt:kelvin_gloss.cc")
    if not is_riscv_test:
        srcs += [
            "//crt:crt.S",
            "//crt:kelvin_start.S",
        ]

    kelvin_binary_impl(
        name = name,
        srcs = srcs,
        linker_script = "//crt:kelvin.ld",
        **kwargs
    )

    # Need to create the following filegroups to make the output discoverable.
    native.filegroup(
        name = "{}.bin".format(name),
        srcs = [name],
        output_group = "bin_file",
    )
    native.filegroup(
        name = "{}.elf".format(name),
        srcs = [name],
        output_group = "elf_file",
    )
