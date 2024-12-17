#ifndef _RKNN_DEMO_PPOCRREC_H_
#define _RKNN_DEMO_PPOCRREC_H_

#include "rknn/rknn_api.h"
#include <string.h>

#define MODEL_OUT_CHANNEL 6625

typedef struct {
    rknn_context rknn_ctx;
    rknn_input_output_num io_num;
    rknn_tensor_attr* input_attrs;
    rknn_tensor_attr* output_attrs;
    int model_channel;
    int model_width;
    int model_height;
} rknn_app_context_t;

typedef struct ppocr_rec_result
{
    char str[512];                                                    // text content
    int str_size;                                                          // text length
    float score;                                                           // text score
    
} ppocr_rec_result;

// int init_ppocr_rec_model(char* model, rknn_app_context_t* app_ctx, int &model_len, rknn_context &ctx);

// int release_ppocr_rec_model(rknn_app_context_t* app_ctx);

// int inference_ppocr_rec_model(rknn_app_context_t* app_ctx, void *buf, ppocr_rec_result* out_result);

int rec_postprocess(float* out_data, int out_channel, int out_seq_len, ppocr_rec_result* text);

#endif //_RKNN_DEMO_PPOCRREC_H_
