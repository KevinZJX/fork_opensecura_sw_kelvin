load(
    "@crt//config:features.bzl",
    "CPP_ALL_COMPILE_ACTIONS",
    "C_ALL_COMPILE_ACTIONS",
    "LD_ALL_ACTIONS",
    "feature",
    "feature_set",
    "flag_group",
    "flag_set",
)

package(default_visibility = ["//visibility:public"])

feature(
    name = "architecture",
    enabled = True,
    flag_sets = [
        flag_set(
            actions = CPP_ALL_COMPILE_ACTIONS + C_ALL_COMPILE_ACTIONS + LD_ALL_ACTIONS,
            flag_groups = [
                flag_group(
                    flags = [
                        "-march=ARCHITECTURE",
                        "-mabi=ABI",
                        "-mcmodel=CMODEL",
                    ],
                ),
            ],
        ),
    ],
)

feature(
    name = "sys_spec",
    enabled = True,
    flag_sets = [
        flag_set(
            actions = CPP_ALL_COMPILE_ACTIONS,
            flag_groups = [
                flag_group(
                    flags = [
                        "-march=ARCHITECTURE",
                        "-mabi=ABI",
                        "-mcmodel=CMODEL",
                        "-nostdlib",
                    ],
                ),
            ],
        ),
        flag_set(
            actions = LD_ALL_ACTIONS,
            flag_groups = [
                flag_group(
                    flags = [
                        "--specs=nano.specs",
                        "-lm_nano",
                        "-lc_nano",
                        "-lg_nano",
                        "-lgcc",
                        "-nostartfiles",
                    ],
                ),
            ],
        ),
    ],
)

feature_set(
    name = "rv32im",
    feature = [
        ":architecture",
        ":sys_spec",
        "@crt//features/embedded:cc_constructor_destructor",
        "@crt//features/embedded:exceptions",
        "@crt//features/embedded:runtime_type_information",
        "@crt//platforms/riscv32/features:all_warnings_as_errors",
        "@crt//platforms/riscv32/features:fastbuild",
        "@crt//features/common:includes",
        "@crt//features/common:all_warnings",
        "@crt//features/common:all_warnings_as_errors",
        "@crt//features/common:reproducible",
        # TODO(atv): It would be nice to have the feature, but for now enabling
        # this creates the wrong program.
        # "@crt//features/common:symbol_garbage_collection",
    ],
)
