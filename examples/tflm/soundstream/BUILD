load("//build_tools/bazel:kelvin.bzl", "generate_cc_arrays", "kelvin_binary")

package(default_visibility = ["//visibility:public"])

kelvin_binary(
    name = "soundstream",
    srcs = [
        "best_of_times_s16_decoded.cc",
        "best_of_times_s16_encoded.cc",
        "best_of_times_s16_wav.cc",
        "decoder_non_stream_q16x8_b64_io_int16_tflite.cc",
        "encoder_non_stream_q16x8_b64_io_int16_tflite.cc",
        "soundstream.cc",
    ],
    hdrs = [
        "best_of_times_s16_decoded.h",
        "best_of_times_s16_encoded.h",
        "best_of_times_s16_wav.h",
        "decoder_non_stream_q16x8_b64_io_int16_tflite.h",
        "encoder_non_stream_q16x8_b64_io_int16_tflite.h",
    ],
    tags = ["manual"],
    deps = [
        "//crt:crt_header",
        "@tflite-micro//tensorflow/lite/micro:micro_framework",
        "@tflite-micro//tensorflow/lite/micro:system_setup",
    ],
)

generate_cc_arrays(
    name = "decoder_non_stream_q16x8_b64_io_int16_tflite_cc",
    src = "@ml-models//:quant_models/_decoder_non_stream_q16x8_b64_io_int16.tflite",
    out = "decoder_non_stream_q16x8_b64_io_int16_tflite.cc",
    tags = ["manual"],
)

generate_cc_arrays(
    name = "decoder_non_stream_q16x8_b64_io_int16_tflite_h",
    src = "@ml-models//:quant_models/_decoder_non_stream_q16x8_b64_io_int16.tflite",
    out = "decoder_non_stream_q16x8_b64_io_int16_tflite.h",
    tags = ["manual"],
)

generate_cc_arrays(
    name = "encoder_non_stream_q16x8_b64_io_int16_tflite_cc",
    src = "@ml-models//:quant_models/_encoder_non_stream_q16x8_b64_io_int16.tflite",
    out = "encoder_non_stream_q16x8_b64_io_int16_tflite.cc",
    tags = ["manual"],
)

generate_cc_arrays(
    name = "encoder_non_stream_q16x8_b64_io_int16_tflite_h",
    src = "@ml-models//:quant_models/_encoder_non_stream_q16x8_b64_io_int16.tflite",
    out = "encoder_non_stream_q16x8_b64_io_int16_tflite.h",
    tags = ["manual"],
)

generate_cc_arrays(
    name = "best_of_times_s16_wav_cc",
    src = "best_of_times_s16.wav",
    out = "best_of_times_s16_wav.cc",
)

generate_cc_arrays(
    name = "best_of_times_s16_wav_h",
    src = "best_of_times_s16.wav",
    out = "best_of_times_s16_wav.h",
)

generate_cc_arrays(
    name = "best_of_times_s16_encoded_cc",
    src = "best_of_times_s16_encoded.raw",
    out = "best_of_times_s16_encoded.cc",
)

generate_cc_arrays(
    name = "best_of_times_s16_encoded_h",
    src = "best_of_times_s16_encoded.raw",
    out = "best_of_times_s16_encoded.h",
)

generate_cc_arrays(
    name = "best_of_times_s16_decoded_cc",
    src = "best_of_times_s16_decoded.raw",
    out = "best_of_times_s16_decoded.cc",
)

generate_cc_arrays(
    name = "best_of_times_s16_decoded_h",
    src = "best_of_times_s16_decoded.raw",
    out = "best_of_times_s16_decoded.h",
)
