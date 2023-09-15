load("//build_tools/bazel:kelvin.bzl", "generate_cc_arrays", "kelvin_binary")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "soundstream",
    srcs = [
        "decoder.cc",
        "decoder_non_stream_q16x8_b64_io_int16_tflite.cc",
        "decoder_non_stream_q16x8_b64_io_int16_tflite.h",
        "decoder_streaming_q16x8_b64_io_int16_tflite.cc",
        "decoder_streaming_q16x8_b64_io_int16_tflite.h",
        "encoder.cc",
        "encoder_non_stream_q16x8_b64_io_int16_tflite.cc",
        "encoder_non_stream_q16x8_b64_io_int16_tflite.h",
        "encoder_streaming_q16x8_b64_io_int16_tflite.cc",
        "encoder_streaming_q16x8_b64_io_int16_tflite.h",
    ],
    hdrs = [
        "decoder.h",
        "encoder.h",
    ],
    tags = ["manual"],
    deps = [
        "@tflite-micro//tensorflow/lite/micro:micro_framework",
        "@tflite-micro//tensorflow/lite/micro:recording_allocators",
    ],
)

kelvin_binary(
    name = "soundstream_decoder_non_streaming",
    srcs = [
        "soundstream_decoder.cc",
    ],
    tags = ["manual"],
    deps = [
        ":soundstream",
        "//crt:crt",
    ],
)

kelvin_binary(
    name = "soundstream_decoder_streaming",
    srcs = [
        "soundstream_decoder.cc",
    ],
    copts = ["-DSTREAMING"],
    tags = ["manual"],
    deps = [
        ":soundstream",
        "//crt:crt",
    ],
)

kelvin_binary(
    name = "soundstream_encoder_non_streaming",
    srcs = [
        "soundstream_encoder.cc",
    ],
    tags = ["manual"],
    deps = [
        ":soundstream",
        "//crt:crt",
    ],
)

kelvin_binary(
    name = "soundstream_encoder_streaming",
    srcs = [
        "soundstream_encoder.cc",
    ],
    copts = ["-DSTREAMING"],
    tags = ["manual"],
    deps = [
        ":soundstream",
        "//crt:crt",
    ],
)

kelvin_binary(
    name = "soundstream_e2e_non_streaming",
    srcs = [
        "best_of_times_s16_decoded.cc",
        "best_of_times_s16_encoded.cc",
        "best_of_times_s16_wav.cc",
        "soundstream_e2e.cc",
    ],
    hdrs = [
        "best_of_times_s16_decoded.h",
        "best_of_times_s16_encoded.h",
        "best_of_times_s16_wav.h",
    ],
    tags = ["manual"],
    deps = [
        ":soundstream",
        "//crt:crt",
    ],
)

kelvin_binary(
    name = "soundstream_e2e_streaming",
    srcs = [
        "best_of_times_s16_decoded_streaming.cc",
        "best_of_times_s16_encoded_streaming.cc",
        "best_of_times_s16_wav.cc",
        "soundstream_e2e.cc",
    ],
    hdrs = [
        "best_of_times_s16_decoded_streaming.h",
        "best_of_times_s16_encoded_streaming.h",
        "best_of_times_s16_wav.h",
    ],
    copts = ["-DSTREAMING"],
    tags = ["manual"],
    deps = [
        ":soundstream",
        "//crt:crt",
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
    name = "decoder_streaming_q16x8_b64_io_int16_tflite_cc",
    src = "@ml-models//:quant_models/_decoder_streaming_q16x8_b64_io_int16.tflite",
    out = "decoder_streaming_q16x8_b64_io_int16_tflite.cc",
    tags = ["manual"],
)

generate_cc_arrays(
    name = "decoder_streaming_q16x8_b64_io_int16_tflite_h",
    src = "@ml-models//:quant_models/_decoder_streaming_q16x8_b64_io_int16.tflite",
    out = "decoder_streaming_q16x8_b64_io_int16_tflite.h",
    tags = ["manual"],
)

generate_cc_arrays(
    name = "encoder_streaming_q16x8_b64_io_int16_tflite_cc",
    src = "@ml-models//:quant_models/_encoder_streaming_q16x8_b64_io_int16.tflite",
    out = "encoder_streaming_q16x8_b64_io_int16_tflite.cc",
    tags = ["manual"],
)

generate_cc_arrays(
    name = "encoder_streaming_q16x8_b64_io_int16_tflite_h",
    src = "@ml-models//:quant_models/_encoder_streaming_q16x8_b64_io_int16.tflite",
    out = "encoder_streaming_q16x8_b64_io_int16_tflite.h",
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

generate_cc_arrays(
    name = "best_of_times_s16_encoded_streaming_cc",
    src = "best_of_times_s16_encoded_streaming.raw",
    out = "best_of_times_s16_encoded_streaming.cc",
)

generate_cc_arrays(
    name = "best_of_times_s16_encoded_streaming_h",
    src = "best_of_times_s16_encoded_streaming.raw",
    out = "best_of_times_s16_encoded_streaming.h",
)

generate_cc_arrays(
    name = "best_of_times_s16_decoded_streaming_cc",
    src = "best_of_times_s16_decoded_streaming.raw",
    out = "best_of_times_s16_decoded_streaming.cc",
)

generate_cc_arrays(
    name = "best_of_times_s16_decoded_streaming_h",
    src = "best_of_times_s16_decoded_streaming.raw",
    out = "best_of_times_s16_decoded_streaming.h",
)
