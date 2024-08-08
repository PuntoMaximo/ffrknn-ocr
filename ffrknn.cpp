#include <ffrknn.h>
#include <SDL_syswm.h>
#include <sdlwidget.h>
#include <iostream>
#include <stdlib.h>

#include <errno.h>
#include <fcntl.h>
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>
#include <thread>

#ifndef DRM_FORMAT_NV12_10
#define DRM_FORMAT_NV12_10 fourcc_code('N', 'A', '1', '2')
#endif

#ifndef DRM_FORMAT_NV15
#define DRM_FORMAT_NV15 fourcc_code('N', 'A', '1', '5')
#endif

void rknnInference(char* labelsListFile);
int ocrDetector(void *resize_buf_crop);
void initPPOCR();

int read_data_from_file(const char *path, char **out_data);
unsigned char *load_data(FILE *fp, size_t ofst, size_t sz);
unsigned char *load_model(char *filename, int *model_size);

std::thread rknn_thread;
std::thread rknn_ocr_thread;
bool default_Settings = true;

/* --- RKNN --- */
int channel = 3;
int m_width = 640;
int m_height = 640;
unsigned char *model_data;
int model_data_size {};
unsigned char *model_data_ocr;
int model_data_size_ocr {};
char *model_name = nullptr;
float scale_w = 1.0f;
float scale_h = 1.0f;
detect_result_group_t detect_result_group;
ppocr_rec_result results;
char score_result[64];
std::vector<float> out_scales;
std::vector<int32_t> out_zps;
rknn_context ctx;
rknn_context ctx_ocr;
rknn_input_output_num io_num;
rknn_input_output_num io_num_ocr;
rknn_input inputs[1];
rknn_input inputs_ocr[1];
rknn_tensor_attr output_attrs[256];
rknn_tensor_attr output_attrs_ocr[256];
const float nms_threshold = NMS_THRESH;
const float box_conf_threshold = BOX_THRESH;
int threadinit {};
cv::Mat src_img;
im_rect crop_rect;
bool ppocrInfer = false;

/* --- SDL --- */
int accur {20};
int alphablend {45};
int frameSize_texture {};
int frameSize_rknn {};
void *resize_buf = nullptr;
void *resize_buf_crop = nullptr;
void *texture_dst_buf = nullptr;
SDL_SysWMinfo info;
Uint32 format;

int screen_width = 1015;
int screen_height = 571;
unsigned int frame_width = 1920;
unsigned int frame_height = 1080;
int v4l2 {};
int rtsp {};
int rtmp {};
int http {};
char *pixel_format;
char *sensor_frame_size;
char *sensor_frame_rate;

/* --- SDL FontCache --- */
FC_Font *font_large = nullptr;

void ffrknn::Init() {

    // default_Settings = defaultSettings();

    char *vname = (char*)"v4l2";

    v4l2 = !strncasecmp(vname, "v4l2", 4);
    // rtsp = !strncasecmp(vname, "rtsp", 4);
    video_name = (char*)"/dev/video0";
    pixel_format = (char*)"nv12";

    sensor_frame_size = (char*)"1920x1080";

    if (!video_name) {
        fprintf(stderr, "No stream to play! Please pass an input.\n");
        return;
    }

    // Creating SDL_Font
    font_small = FC_CreateFont();

    if (!font_small) {
        fprintf(stderr, "No small ttf can be created.\n");
        return;
    }

    std::string image_path = "/usr/share/img.jpg";

    src_img = cv::imread(image_path);
    if (src_img.empty()) {
        std::cerr << "Error loading image: " << image_path << std::endl;
        return;
    }

    // Initializes YOLOv5 Model
    initYOLOV5();

    memset(&crop_rect, 0, sizeof(crop_rect));

    // im_rect values for RGA_CROP.
    crop_rect.x = 200;
    crop_rect.y = 100;
    crop_rect.width = 320;
    crop_rect.height = 48;

    input_ctx = avformat_alloc_context();
    if (!input_ctx) {
        av_log(0, AV_LOG_ERROR, "Cannot allocate input format (Out of memory?)\n");
        return;
    }

    av_dict_set(&opts, "num_capture_buffers", "128", 0);

    if (rtsp) {
        // av_dict_set(&opts, "rtsp_transport", "tcp", 0);
        av_dict_set(&opts, "rtsp_flags", "prefer_tcp", 0);
    }

    if (v4l2) {
        avdevice_register_all();
        ifmt = av_find_input_format("video4linux2");

        if (!ifmt) {
            av_log(0, AV_LOG_ERROR, "Cannot find input format: v4l2\n");
            return;
        }
        input_ctx->flags |= AVFMT_FLAG_NONBLOCK;

        if (pixel_format) {
            av_dict_set(&opts, "input_format", pixel_format, 0);
        }

        if (sensor_frame_size)
            av_dict_set(&opts, "video_size", sensor_frame_size, 0);

        if (sensor_frame_rate)
            av_dict_set(&opts, "framerate", sensor_frame_rate, 0);
    }

    if (rtmp) {
        ifmt = av_find_input_format("flv");
        if (!ifmt) {
            av_log(0, AV_LOG_ERROR, "Cannot find input format: flv\n");
            return;
        }
        av_dict_set(&opts, "fflags", "nobuffer", 0);
    }

    if (http) {
        av_dict_set(&opts, "fflags", "nobuffer", 0);
    }

    if (avformat_open_input(&input_ctx, video_name, ifmt, &opts) != 0) {
        av_log(0, AV_LOG_ERROR, "Cannot open input file '%s'\n", video_name);
        avformat_close_input(&input_ctx);
        return;
    }

    if (avformat_find_stream_info(input_ctx, NULL) < 0) {
        av_log(0, AV_LOG_ERROR, "Cannot find input stream information.\n");
        avformat_close_input(&input_ctx);
        return;
    }

    /* find the video stream information */
    ret = av_find_best_stream(input_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, &codec, 0);
    if (ret < 0) {
        av_log(0, AV_LOG_ERROR, "Cannot find a video stream in the input file\n");
        avformat_close_input(&input_ctx);
        return;
    }
    video_stream = ret;

    /* find the video decoder: ie: h264_rkmpp / h264_rkmpp_decoder */
    codecpar = input_ctx->streams[video_stream]->codecpar;
    if (!codecpar) {
        av_log(0, AV_LOG_ERROR, "Unable to find stream!\n");
        avformat_close_input(&input_ctx);
        return;
    }

#if 0
    if (codecpar->codec_id != AV_CODEC_ID_H264) {
        av_log(0, AV_LOG_ERROR, "H264 support only!\n");
        avformat_close_input(&input_ctx);
        return -1;
    }
#endif

    codec = avcodec_find_decoder(codecpar->codec_id);
    if (!codec) {
        av_log(0, AV_LOG_ERROR, "Codec not found!\n");
        avformat_close_input(&input_ctx);
        return;
    }

    codec_ctx = avcodec_alloc_context3(codec);
    if (!codec_ctx) {
        av_log(0, AV_LOG_ERROR, "Could not allocate video codec context!\n");
        avformat_close_input(&input_ctx);
        return;
    }

    video = input_ctx->streams[video_stream];
    if (avcodec_parameters_to_context(codec_ctx, video->codecpar) < 0) {
        av_log(0, AV_LOG_ERROR, "Error with the codec!\n");
        avformat_close_input(&input_ctx);
        avcodec_free_context(&codec_ctx);
        return;
    }

    codec_ctx->pix_fmt = AV_PIX_FMT_NV12;
    codec_ctx->coded_height = frame_height;
    codec_ctx->coded_width = frame_width;
    codec_ctx->get_format = nullptr;
    codec_ctx->skip_alpha = 1;

#if 0
    while (dict = av_dict_get(opts, "", dict, AV_DICT_IGNORE_SUFFIX)) {
        fprintf(stderr, "dict: %s -> %s\n", dict->key, dict->value);
    }
#endif

    /* open it */
    if (avcodec_open2(codec_ctx, codec, &opts) < 0) {
        av_log(0, AV_LOG_ERROR, "Could not open codec!\n");
        avformat_close_input(&input_ctx);
        avcodec_free_context(&codec_ctx);
        return;
    }

    av_dict_free(&opts);

    frame = av_frame_alloc();
    if (!frame) {
        fprintf(stderr, "Could not allocate video frame\n");
        avformat_close_input(&input_ctx);
        avcodec_free_context(&codec_ctx);
        return;
    }

    SDL_VERSION(&sdl_compiled);
    SDL_GetVersion(&sdl_linked);
    SDL_Log("SDL: compiled with=%d.%d.%d linked against=%d.%d.%d",
            sdl_compiled.major, sdl_compiled.minor, sdl_compiled.patch,
            sdl_linked.major, sdl_linked.minor, sdl_linked.patch);

    if (SDL_Init(SDL_INIT_EVERYTHING) < 0) {
        SDL_Log("SDL_Init failed (%s)", SDL_GetError());
        return;
    }

    printf("Creating SDL window and renderer\n");

    window = SDL_CreateWindowFrom(reinterpret_cast<void*>(winId()));
    // window = SDL_CreateWindow("ff-rknn-v4l2-thread", 0, 0, screen_width, screen_height, 0 | SDL_WINDOW_OPENGL | SDL_WINDOW_BORDERLESS | SDL_WINDOW_ALWAYS_ON_TOP);
    if(window == nullptr) {
        std::cout << "Can't create window: " << SDL_GetError() << std::endl;
        return;
    }

    SDL_SetHint(SDL_HINT_RENDER_SCALE_QUALITY, "linear");

    renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
    if(renderer == nullptr) {
        std::cout << "Can't create renderer: " << SDL_GetError() << std::endl;
        return;
    }

    texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGB24, SDL_TEXTUREACCESS_STREAMING,
                                1015, 571);
    if (!texture) {
        av_log(NULL, AV_LOG_FATAL, "Failed to create texturer: %s", SDL_GetError());
    }

    if (alphablend) {
        SDL_SetRenderDrawBlendMode(renderer, SDL_BLENDMODE_BLEND);
    }

    format = SDL_PIXELFORMAT_RGB24;

    frameSize_rknn = m_width * m_height * channel;
    resize_buf = calloc(1, frameSize_rknn);

    frameSize_texture = screen_width * screen_height * channel;
    texture_dst_buf = calloc(1, frameSize_texture);

    if (!resize_buf || !texture_dst_buf) {
        av_log(NULL, AV_LOG_FATAL, "Failed to create texture buf: %dx%d",
               screen_width, screen_height);
    }

    if (!FC_LoadFont(font_small, renderer, "/usr/share/fonts/FreeSans.ttf", 14, FC_MakeColor(255, 255, 255, 255), TTF_STYLE_NORMAL))
        std::cerr << "LOAD FONT ERROR" << std::endl;
    std::cout << "FONT LOAD SUCCESS." << std::endl;

}

void ffrknn::Update() {

    if ((ret = av_read_frame(input_ctx, &pkt)) < 0) {
        if (ret == AVERROR(EAGAIN)) {
        }
    }
    if (video_stream == pkt.stream_index && pkt.size > 0) {

        ret = decode_and_display(codec_ctx, frame, &pkt);
    }
    av_packet_unref(&pkt);

    while (SDL_PollEvent(&event)) {

        switch (event.type) {
            case SDL_QUIT: {



                if (input_ctx)
                    avformat_close_input(&input_ctx);
                if (codec_ctx)
                    avcodec_free_context(&codec_ctx);
                if (frame) {
                    av_frame_free(&frame);
                }

                if (threadinit == 1) {

                    if (rknn_thread.joinable())
                        rknn_thread.join();

                    if (ctx) {
                        rknn_destroy(ctx);
                    }

                    if (ctx_ocr){
                        rknn_destroy(ctx_ocr);
                    }

                    deinitPostProcess();

                    if (model_data) {
                        free(model_data);
                    }

                }

                if (font_small) {
                    FC_FreeFont(font_small);
                }

                SDLWidget::~SDLWidget();

            }
        }
    }
}

void ffrknn::OnResize(int width, int height) {

    // Ajustar las dimensiones de la ventana SDL al nuevo tamaño del widget
    std::cout << "SDL Window resize.." << std::endl;
    if (window) {
        SDL_SetWindowSize(window, width, height);
    }

}

int ffrknn::decode_and_display(AVCodecContext *dec_ctx, AVFrame *frame, AVPacket *pkt) {

    AVDRMFrameDescriptor *desc;
    AVDRMLayerDescriptor *layer;
    unsigned int drm_format;
    RgaSURF_FORMAT src_format;
    RgaSURF_FORMAT dst_format;
    int hStride, wStride;
    SDL_Rect rect;
    int ret;

    ret = avcodec_send_packet(dec_ctx, pkt);

    if (ret < 0) {
        fprintf(stderr, "Error sending a packet for decoding\n");
        return ret;
    }
    ret = 0;

    if (threadinit == 0) {
        // labelsListFile = (char*)"/usr/share/coco_80_labels_list_original.txt";
        labelsListFile = (char*)"/usr/share/coco_80_labels_list_placas.txt";

        if (!default_Settings) {
            if (selectLanguage())
                labelsListFile = (char*)"/usr/share/coco_80_labels_list_original.txt";
            else
                labelsListFile = (char*)"/usr/share/coco_80_labels_list.txt";
        }

        rknn_thread = std::thread(rknnInference, std::ref(labelsListFile));
        rknn_thread.detach();
        threadinit = 1;
    }
    while (ret >=0) {

        ret = avcodec_receive_frame(dec_ctx, frame);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
            break;
        } else if (ret < 0) {
            fprintf(stderr, "Error during decoding!\n");
            return ret;
        }
        desc = (AVDRMFrameDescriptor *)frame->data[0];
        layer = &desc->layers[0];
        if (desc && layer) {

            wStride = frame_width;
            hStride = frame_height;
            drm_format = layer->format;
            src_format = (RgaSURF_FORMAT)drm_get_rgaformat(DRM_FORMAT_NV12);

            /* ------------ RKNN ----------- */

            drm_rga_buf(frame->width, frame->height, wStride, hStride, desc->objects[0].fd, src_format,
                        screen_width, screen_height, RK_FORMAT_BGR_888,
                        frameSize_texture, (char *)texture_dst_buf, (char *) frame->data[0]);

            drm_rga_buf(frame->width, frame->height, frame->width, frame->height, desc->objects[0].fd, src_format,
                        m_width, m_height, RK_FORMAT_BGR_888,
                        frameSize_rknn, (char *)resize_buf, (char *) frame->data[0]);

            displayTexture(texture_dst_buf);
        }
    }

    return 0;
}

uint32_t ffrknn::drm_get_rgaformat(uint32_t drm_fmt) {

    switch (drm_fmt) {
    case DRM_FORMAT_NV12:
        return RK_FORMAT_YCbCr_420_SP;
    case DRM_FORMAT_NV12_10:
        return RK_FORMAT_YCbCr_420_SP_10B;
    case DRM_FORMAT_NV15:
        return RK_FORMAT_YCbCr_420_SP_10B;
    case DRM_FORMAT_NV16:
        return RK_FORMAT_YCbCr_422_SP;
    case DRM_FORMAT_YUYV:
        return RK_FORMAT_YUYV_422;
    case DRM_FORMAT_UYVY:
        return RK_FORMAT_UYVY_422;

    default:
        return 0;
    }
}

// int ffrknn::drm_rga_buf(int src_Width, int src_Height, int wStride, int hStride, int src_fd,
//                         int src_format, int dst_Width, int dst_Height,
//                         int dst_format, int frameSize, char *buf, char *viraddr)
// {
//     int ret {};
//     int dst_buf_size {};

//     rga_info_t src;
//     rga_info_t dst;

//     rga_buffer_t src_img, dst_img;

//     memset(&src_img, 0, sizeof(src_img));
//     memset(&dst_img, 0, sizeof(dst_img));

//     // dst_buf_size = crop_rect.width * crop_rect.height * 3;

//     // // Asigna memoria para el buffer recortado
//     // resize_buf_crop = (char*)malloc(dst_buf_size);

//     // if (!resize_buf_crop) {
//     //     fprintf(stderr, "Failed to allocate memory for resize_buf_crop\n");
//     //     return -1;
//     // }

//     // memset(resize_buf_crop, 0x80, dst_buf_size);

//     // Configura las estructuras RGA
//     memset(&src, 0, sizeof(rga_info_t));
//     src.fd = -1;
//     src.virAddr = viraddr;
//     src.mmuFlag = 1;

//     memset(&dst, 0, sizeof(rga_info_t));
//     dst.fd = -1;
//     dst.virAddr = buf;
//     dst.mmuFlag = 1;

//     rga_set_rect(&src.rect, 0, 0, src_Width, src_Height, wStride, hStride, src_format);
//     rga_set_rect(&dst.rect, 0, 0, dst_Width, dst_Height, dst_Width, dst_Height, dst_format);

//     // Ejecuta la operación de blit
//     ret = c_RkRgaBlit(&src, &dst, NULL);

//     // // Configura los buffers de origen y destino para RGA
//     // src_img = wrapbuffer_virtualaddr((void*)resize_buf, dst_Width, dst_Height, src_format);
//     // dst_img = wrapbuffer_virtualaddr((void*)resize_buf_crop, crop_rect.width, crop_rect.height, dst_format);

//     // // Verifica los buffers de origen y destino
//     // ret = imcheck(src_img, dst_img, {}, {});
//     // if (IM_STATUS_NOERROR != ret) {
//     //     printf("%d, check error! %s\n", __LINE__, imStrError((IM_STATUS)ret));
//     //     free(resize_buf_crop);
//     //     return -1;
//     // }

//     // // Realiza el recorte
//     // ret = imcrop(src_img, dst_img, crop_rect);
//     // if (ret == IM_STATUS_SUCCESS) {
//     //     //std::cout << "crop running success!" << std::endl;
//     // } else {
//     //     std::cout << "crop running failed!" << std::endl;
//     //     free(resize_buf_crop);
//     //     return -1;
//     // }

//     // // Guardar el buffer recortado como una imagen para verificar
//     // cv::Mat cropped_image(48, 320, CV_8UC3, resize_buf);
//     // cv::imwrite("/usr/share/cropped_image.png", cropped_image);
//     // std::cout << "Cropped image saved as cropped_image.png" << std::endl;

//     // Libera la memoria del buffer recortado
//     free(resize_buf_crop);

//     return ret;
// }

void ffrknn::save_and_display_crop_with_opencv(char* buffer, int width, int height) {
    // Crear un objeto Mat desde el buffer
    // cv::Mat img(height, width, CV_8UC3, buffer);
    cv::Mat img(48, 320, CV_32FC3, resize_buf_crop);

    // Guardar la imagen en un archivo
    cv::imwrite("crop_output.png", img);

    img.release();
}

int ffrknn::drm_rga_buf(int src_Width, int src_Height, int wStride, int hStride, int src_fd,
                        int src_format, int dst_Width, int dst_Height,
                        int dst_format, int frameSize, char *buf, char *viraddr)
{
    rga_info_t src;
    rga_info_t dst;
    rga_info_t crop_dst;
    int ret;
    int dst_buf_size {};

    // int hStride = (src_Height + 15) & (~15);
    // int wStride = (src_Width + 15) & (~15);
    // int dhStride = (dst_Height + 15) & (~15);
    // int dwStride = (dst_Width + 15) & (~15);

    // dst_buf_size = crop_rect.width * crop_rect.height * 3;

    // resize_buf_crop = (char*)malloc(dst_buf_size);
    // if (!resize_buf_crop) {
    //     return -1;
    // }

    memset(&src, 0, sizeof(rga_info_t));
    src.fd = -1;
    src.virAddr = viraddr;
    src.mmuFlag = 1;

    memset(&dst, 0, sizeof(rga_info_t));
    dst.fd = -1;
    dst.virAddr = buf;
    dst.mmuFlag = 1;

    rga_set_rect(&src.rect, 0, 0, src_Width, src_Height, wStride, hStride,
                 src_format);
    rga_set_rect(&dst.rect, 0, 0, dst_Width, dst_Height, dst_Width, dst_Height,
                 dst_format);

    ret = c_RkRgaBlit(&src, &dst, NULL);

    if (ret != 0) {
        return ret;
    }

    // memset(&crop_dst, 0, sizeof(rga_info_t));
    // crop_dst.fd = -1;
    // crop_dst.virAddr = resize_buf_crop;
    // crop_dst.mmuFlag = 1;

    // rga_set_rect(&dst.rect, crop_rect.x, crop_rect.y, crop_rect.width, crop_rect.height, dst_Width, dst_Height, dst_format);
    // rga_set_rect(&crop_dst.rect, 0, 0, crop_rect.width, crop_rect.height, crop_rect.width, crop_rect.height, dst_format);

    // ret = c_RkRgaBlit(&dst, &crop_dst, NULL);

    // // if (ret == 0) {
    // //     save_and_display_crop_with_opencv((char*)resize_buf_crop, crop_rect.width, crop_rect.height);
    // // }

    // free(resize_buf_crop);
    // return ret;
}

void ffrknn::displayTexture(void *imageData)
{
    unsigned char *texture_data = NULL;
    int texture_pitch = 0;

    SDL_LockTexture(texture, 0, (void **)&texture_data, &texture_pitch);
    memcpy(texture_data, (void *)imageData, frameSize_texture);
    SDL_UnlockTexture(texture);
    SDL_RenderCopy(renderer, texture, NULL, NULL);

    // Draw Objects
    char text[512];
    SDL_Rect rect;
    SDL_Rect rect_;
    SDL_Rect rect_bar;
    unsigned int obj;
    int accur_obj;
    int clr;

    // rect_.x = 200;
    // rect_.y = 100;
    // rect_.w = 320;
    // rect_.h = 48;

    // SDL_RenderDrawRect(renderer, &rect_);

    for (int i = 0; i < detect_result_group.count; i++) {
        detect_result_t *det_result = &(detect_result_group.results[i]);
        // sprintf(text, "%s %.1f%%", det_result->name, det_result->prop * 100);

        if (accur) {
            accur_obj = (int)(det_result->prop * 100.0);
            if (accur_obj < accur) {
                continue;
            }
        }

        rect.x = det_result->box.left;
        rect.y = det_result->box.top;
        rect.w = det_result->box.right - det_result->box.left + 1;
        rect.h = det_result->box.bottom - det_result->box.top + 1;

        std::string detection = det_result->name;

        if (detection == "Placa") {
            ppocrInfer = true;
            sprintf(text, "%s - %s", det_result->name, results.str);
            // crop_rect.x = rect.x;
            // crop_rect.y = rect.y;
            // crop_rect.width = rect.w;
            // crop_rect.height = rect.h;

            crop_rect.x = static_cast<int>(det_result->box.left * scale_w);
            crop_rect.y = static_cast<int>(det_result->box.top * scale_h);
            crop_rect.width = static_cast<int>((det_result->box.right - det_result->box.left + 1) * scale_w);
            crop_rect.height = static_cast<int>((det_result->box.bottom - det_result->box.top + 1) * scale_h);

            // crop_rect.x = scaled_x;
            // crop_rect.y = scaled_y;
            // crop_rect.width = scaled_w;
            // crop_rect.height = scaled_h;

            std::cout << "rectx = " << crop_rect.x << " " << "recty = " << crop_rect.y << "\n"
                      << "rectw = " << crop_rect.width << " " << "recth = " << crop_rect.height << "\n" << std::endl;

            if (crop_rect.x < 0) crop_rect.x = 0;
            if (crop_rect.y < 0) crop_rect.y = 0;
            if (crop_rect.x + crop_rect.width > 640) crop_rect.width = 640 - crop_rect.x;
            if (crop_rect.y + crop_rect.height > 640) crop_rect.height = 640 - crop_rect.y;
        }

        ppocrInfer = false;

        sprintf(text, "%s - %s", det_result->name, results.str);

        if (det_result->name[0] == 'V' && det_result->name[1] == 'e')
            clr = 1;
        else if (det_result->name[0] == 'P' && det_result->name[1] == 'l')
            clr = 2;
        else
            clr = 0;

        if (alphablend) {
            if (clr == 1)
                SDL_SetRenderDrawColor(renderer, 0, 0, 255, alphablend);
            else if (clr == 2)
                SDL_SetRenderDrawColor(renderer, 51, 153, 255, alphablend);
            SDL_RenderFillRect(renderer, &rect);
        }

        if (clr == 1)
            SDL_SetRenderDrawColor(renderer, 0, 0, 255, SDL_ALPHA_OPAQUE);
        else if (clr == 2)
            SDL_SetRenderDrawColor(renderer, 51, 153, 255, SDL_ALPHA_OPAQUE);
        else
            SDL_SetRenderDrawColor(renderer, 0, 0, 255, SDL_ALPHA_OPAQUE);
        SDL_RenderDrawRect(renderer, &rect);

        rect_bar.x = rect.x;
        rect_bar.h = 22;
        rect_bar.w = rect.w;
        if (rect.w < 80)
            rect_bar.h += 16;
        rect_bar.y = rect.y - rect_bar.h;
        SDL_RenderFillRect(renderer, &rect_bar);
        rect_bar.y -= 1;
        FC_DrawBox(font_small, renderer, rect_bar, text);
        // FC_Draw(font_small, renderer, rect.x + 20, rect.y - 40, results.str, 30);
    }

    //rect = FC_Draw(font_small, renderer, 475.5, 520, "VonVision-AI", 20);
    FC_Draw(font_small, renderer, 475.5, 520, "VonVision-AI", 20);
    SDL_RenderPresent(renderer);
}


AVPixelFormat ffrknn::get_format(AVCodecContext *Context, const enum AVPixelFormat *PixFmt)
{
    while (*PixFmt != AV_PIX_FMT_NONE) {
        if (*PixFmt == AV_PIX_FMT_NV12)
            return AV_PIX_FMT_NV12;
        PixFmt++;
    }
    return AV_PIX_FMT_NONE;
}

unsigned char *load_model(char *filename, int *model_size)
{

    FILE *fp;
    unsigned char *data;

    if (!filename)
        return NULL;

    fp = fopen(filename, "rb");
    if (NULL == fp) {
        fprintf(stderr, "Open file %s failed.\n", filename);
        return NULL;
    }

    fseek(fp, 0, SEEK_END);
    int size = ftell(fp);

    data = load_data(fp, 0, size);

    fclose(fp);

    *model_size = size;
    return data;
}

unsigned char *load_data(FILE *fp, size_t ofst, size_t sz)
{
    unsigned char *data;
    int ret;

    data = NULL;

    if (NULL == fp) {
        return NULL;
    }

    ret = fseek(fp, ofst, SEEK_SET);
    if (ret != 0) {
        fprintf(stderr, "blob seek failure.\n");
        return NULL;
    }

    data = (unsigned char *)malloc(sz);
    if (data == NULL) {
        fprintf(stderr, "buffer malloc failure.\n");
        return NULL;
    }
    ret = fread(data, 1, sz, fp);
    return data;
}

void ffrknn::initYOLOV5() {

    // model_name = (char*)"/usr/share/yolov5n.rknn";
    model_name = (char*)"/usr/share/car-license-yolov5n.rknn";

    if (!default_Settings) {
        if (selectModel())
            model_name = (char*)"/usr/share/yolov5n.rknn";
        else
            model_name = (char*)"/usr/share/yolov5s_relu.rknn";
        std::cout << "Inference Accuracy (int) : ";
        std::cin >> accur;
    }

    if (!model_name) {
        fprintf(stderr, "No model to load! Please pass a model.\n");
        return;
    }

    /* Create the neural network */
    model_data_size = 0;
    model_data = load_model(model_name, &model_data_size);

    if (!model_data) {
        fprintf(stderr, "Error locading model: `%s`\n", model_name);
        return;
    }

    fprintf(stderr, "Model: %s - size: %d.\n", model_name, model_data_size);
    ret = rknn_init(&ctx, model_data, model_data_size, 0, NULL);
    if (ret < 0) {
        fprintf(stderr, "rknn_init error ret=%d\n", ret);
        return;
    }

    rknn_sdk_version version;
    ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &version,
                     sizeof(rknn_sdk_version));
    if (ret < 0) {
        fprintf(stderr, "rknn_init error ret=%d\n", ret);
        return;
    }
    fprintf(stderr, "sdk version: %s driver version: %s\n",
            version.api_version,
            version.drv_version);

    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0) {
        fprintf(stderr, "rknn_init error ret=%d\n", ret);
        return;
    }
    fprintf(stderr, "model input num: %d, output num: %d\n",
            io_num.n_input,
            io_num.n_output);

    rknn_tensor_attr input_attrs[io_num.n_input + 1];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++) {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]),
                         sizeof(rknn_tensor_attr));
        if (ret < 0) {
            fprintf(stderr, "rknn_init error ret=%d\n", ret);
            return;
        }
    }

    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++) {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]),
                         sizeof(rknn_tensor_attr));
    }

    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW) {
        channel = input_attrs[0].dims[1];
        m_width = input_attrs[0].dims[2];
        m_height = input_attrs[0].dims[3];
    } else {
        m_width = input_attrs[0].dims[1];
        m_height = input_attrs[0].dims[2];
        channel = input_attrs[0].dims[3];
    }

    fprintf(stderr, "model: %dx%dx%d\n", m_width, m_height, channel);
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = m_width * m_height * channel;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].pass_through = 0;

    // if (model_data) {
    //     free(model_data);
    // }

}

void initPPOCR() {
    int m_width = 320;
    int m_height = 48;

    model_name = (char*)"/usr/share/ppocrv4_rec.rknn";
    if (!model_name) {
        fprintf(stderr, "No model to load! Please pass a model.\n");
        return;
    }

    model_data_ocr = load_model(model_name, &model_data_size_ocr);
    if (!model_data_ocr) {
        fprintf(stderr, "Error loading model: `%s`\n", model_name);
        return;
    }

    //fprintf(stderr, "Model PPOCR: %s - size: %d.\n", model_name, model_data_size_ocr);
    int ret = rknn_init(&ctx_ocr, model_data_ocr, model_data_size_ocr, 0, NULL);
    if (ret < 0) {
        fprintf(stderr, "rknn_init error ret=%d\n", ret);
        return;
    }

    // Get Model Input Output Number
    ret = rknn_query(ctx_ocr, RKNN_QUERY_IN_OUT_NUM, &io_num_ocr, sizeof(io_num_ocr));
    if (ret != RKNN_SUCC) {
        printf("rknn_query fail! ret=%d\n", ret);
        return;
    }
    //printf("model input num: %d, output num: %d\n", io_num_ocr.n_input, io_num_ocr.n_output);

    rknn_tensor_attr input_attrs[io_num_ocr.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));

    for (int i = 0; i < io_num_ocr.n_input; i++) {
        input_attrs[i].index = i;
        ret = rknn_query(ctx_ocr, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret < 0) {
            fprintf(stderr, "rknn_query input attr error ret=%d\n", ret);
            return;
        }
    }

    memset(output_attrs_ocr, 0, sizeof(output_attrs_ocr));
    for (int i = 0; i < io_num_ocr.n_output; i++) {
        output_attrs_ocr[i].index = i;
        ret = rknn_query(ctx_ocr, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs_ocr[i]), sizeof(rknn_tensor_attr));
        if (ret < 0) {
            fprintf(stderr, "rknn_query output attr error ret=%d\n", ret);
            return;
        }
    }

    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW) {
        channel = input_attrs[0].dims[1];
        m_width = input_attrs[0].dims[2];
        m_height = input_attrs[0].dims[3];
    } else {
        m_width = input_attrs[0].dims[1];
        m_height = input_attrs[0].dims[2];
        channel = input_attrs[0].dims[3];
    }

    //fprintf(stderr, "model: %dx%dx%d\n", m_width, m_height, channel);
    memset(inputs_ocr, 0, sizeof(inputs_ocr));

    // Set Input Data
    inputs_ocr[0].index = 0;
    inputs_ocr[0].type = RKNN_TENSOR_FLOAT32;
    inputs_ocr[0].fmt = RKNN_TENSOR_NHWC;
    inputs_ocr[0].size = m_width * m_height * channel * sizeof(float);
    inputs_ocr[0].pass_through = 0;

    if (model_data_ocr) {
        free(model_data_ocr);
    }
}


void rknnInference(char* labelsListFile) {

    int ret {};

    while (true) {
        inputs[0].buf = resize_buf;

        ret = rknn_inputs_set(ctx, io_num.n_input, inputs);
        rknn_output outputs[io_num.n_output];

        memset(outputs, 0, sizeof(outputs));

        for (int i = 0; i < io_num.n_output; i++) {
            outputs[i].want_float = 0;
        }

        ret = rknn_run(ctx, NULL);
        ret = rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);

        // post process
        scale_w = (float)m_width / screen_width;
        scale_h = (float)m_height / screen_height;

        for (int i = 0; i < io_num.n_output; ++i) {
            out_scales.push_back(output_attrs[i].scale);
            out_zps.push_back(output_attrs[i].zp);
        }

        post_process((int8_t *)outputs[0].buf, (int8_t *)outputs[1].buf, (int8_t *)outputs[2].buf,
                     m_height, m_width, box_conf_threshold, nms_threshold,
                     scale_w, scale_h, out_zps, out_scales, &detect_result_group, labelsListFile);

        ret = rknn_outputs_release(ctx, io_num.n_output, outputs);

        // if (ppocrInfer) {
        //     initPPOCR();
        //     ocrDetector(resize_buf);
        // }
    }
}

int ocrDetector(void *resize_buf_crop) {
    int ret;
    rknn_output outputs[1];

    memset(outputs, 0, sizeof(outputs));

    cv::Mat src_img(640, 640, CV_8UC3, resize_buf_crop);

    cv::Rect crop_region(crop_rect.x, crop_rect.y, crop_rect.width, crop_rect.height);
    cv::Mat cropped_img = src_img(crop_region);

    // Redimensionar cropped_img a 320x48
    cv::Mat resized_img;
    cv::resize(cropped_img, resized_img, cv::Size(320, 48), cv::INTER_NEAREST);

    // Normalizar la imagen
    cv::Mat normalized_img;
    resized_img.convertTo(normalized_img, CV_32FC3, 1.0 / 127.5, -1.0);

    // Asignar memoria para el buffer de entrada
    inputs_ocr[0].buf = malloc(inputs_ocr[0].size);
    if (inputs_ocr[0].buf == NULL) {
        printf("Error allocating memory for input buffer\n");
        return -1;
    }

    // Copiar datos de imagen normalizados al buffer de entrada
    memcpy(inputs_ocr[0].buf, normalized_img.data, inputs_ocr[0].size);

    ret = rknn_inputs_set(ctx_ocr, io_num_ocr.n_input, inputs_ocr);
    if (ret < 0) {
        printf("rknn_inputs_set fail! ret=%d\n", ret);
        free(inputs_ocr[0].buf); // Liberar memoria antes de retornar
        return -1;
    }

    ret = rknn_run(ctx_ocr, nullptr);
    if (ret < 0) {
        printf("rknn_run fail! ret=%d\n", ret);
        free(inputs_ocr[0].buf); // Liberar memoria antes de retornar
        return -1;
    }

    // Obtener salida
    int out_len_seq = 320 / 8;
    outputs[0].want_float = 1;
    ret = rknn_outputs_get(ctx_ocr, io_num_ocr.n_output, outputs, NULL);
    if (ret < 0) {
        printf("rknn_outputs_get fail! ret=%d\n", ret);
        free(inputs_ocr[0].buf); // Liberar memoria antes de retornar
        return -1;
    }

    // Post procesamiento
    ret = rec_postprocess((float*)outputs[0].buf, MODEL_OUT_CHANNEL, out_len_seq, &results);

    // Liberar recursos de salida
    rknn_outputs_release(ctx_ocr, io_num_ocr.n_output, outputs);
    free(inputs_ocr[0].buf);
    ret = rknn_destroy(ctx_ocr);

    return ret;
}


bool ffrknn::selectLanguage() {

    // True for English, False for Spanish.
    bool englishOrSpanish = true;

    std::string option;
    std::cout << "::: VonVision-AI :::\n" << std::endl;
    std::cout << "Select Language:\n" << std::endl;
    std::cout << "1. English" << std::endl;
    std::cout << "2. Spanish\n" << std::endl;
    std::cout << "Option: ";
    std::cin >> option;

    if (option == "1") {
        std::cout << "English language set." << std::endl;
        return englishOrSpanish;
    } else if (option == "2") {
        std::cout << "Seleccionado Español como idioma." << std::endl;
        return !englishOrSpanish;
    } else {
        std::cout << "Invalid option. Please try again." << std::endl;
    }

}

bool ffrknn::selectModel() {

    // True for YOLOv5n, False for YOLOv5s.
    bool yolov5nOrYolov5s = true;

    std::string option;
    std::cout << "\nSelect Model:\n" << std::endl;
    std::cout << "1. YOLOv5n" << std::endl;
    std::cout << "2. YOLOv5s\n" << std::endl;
    std::cout << "Option: ";
    std::cin >> option;

    if (option == "1") {
        std::cout << "YOLOv5n set." << std::endl;
        return yolov5nOrYolov5s;
    } else if (option == "2") {
        std::cout << "YOLOv5s set." << std::endl;
        return !yolov5nOrYolov5s;
    } else {
        std::cout << "Invalid option. Please try again." << std::endl;
    }

}

bool ffrknn::defaultSettings() {

    bool defaultSettings = true;

    std::string option;
    std::cout << "\nDefault Settings? : y/n\n" << std::endl;
    std::cout << "Option: ";
    std::cin >> option;

    if (option == "y") {
        return defaultSettings;
    } else if (option == "n") {
        return !defaultSettings;
    } else {
        std::cout << "Invalid option. Please try again." << std::endl;
    }
}
