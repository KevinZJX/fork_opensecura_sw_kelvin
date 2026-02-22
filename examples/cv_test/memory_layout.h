#ifndef _MEMORY_LAYOUT_H_
#define _MEMORY_LAYOUT_H_

#include <stddef.h>
#include <stdint.h>

extern char __model_input_start__[];
extern char __model_input_end__[];

extern char __model_output_start__[];
extern char __model_output_end__[];

extern char __input_header_start__[];
extern char __input_header_end__[];

extern char __output_header_start__[];
extern char __output_header_end__[];

#define MODEL_INPUT_SIZE ((size_t)(__model_input_end__ - __model_input_start__))
#define MODEL_OUTPUT_SIZE ((size_t)(__model_output_end__ - __model_output_start__))
#define MODEL_OUTPUT_HEADER_SIZE ((size_t)(__output_header_end__ - __output_header_start__))
#define MODEL_INPUT_HEADER_SIZE ((size_t)(__input_header_end__ - __input_header_start__))

static inline void* get_model_input_buffer(void) {
    return __model_input_start__;
}

static inline void* get_model_output_buffer(void) {
    return __model_output_start__;
}

static inline void* get_model_output_header_buffer(void) {
    return __output_header_start__;
}

static inline void* get_model_input_header_buffer(void) {
    return __input_header_start__;
}

extern uint32_t _ret;

#endif
