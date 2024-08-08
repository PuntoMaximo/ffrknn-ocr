#pragma once
#include <sdlwidget.h>
#include <rknn/rknn_api.h>
#include <QPushButton>
#include <ppocr_rec.h>

#include <rga/RgaUtils.h>
#include <rga/im2d.h>
#include <rga/RgaApi.h>
#include <rga/rga.h>

#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>

class ffrknn : public SDLWidget
{
    Q_OBJECT

public:
    ffrknn(QWidget* parent = nullptr)
        : SDLWidget(parent) {}
    ~ffrknn() override = default;

    bool isSpanishLang = true;
    bool isSpanishLang2 = true;

private:
    void Init() override;
    void Update() override;
    void OnResize(int w, int h) override;

    // --------- SDL
    SDL_Event event;
    SDL_version sdl_compiled;
    SDL_version sdl_linked;
    int decode_and_display(AVCodecContext *dec_ctx, AVFrame *frame, AVPacket *pkt) override;
    void displayTexture(void *imageData);

    // --------- FFmpeg
    AVPixelFormat get_format(AVCodecContext *Context, const enum AVPixelFormat *PixFmt);
    uint32_t drm_get_rgaformat(uint32_t drm_fmt);
    AVFormatContext *input_ctx = nullptr;
    AVStream *video = nullptr;
    int video_stream, ret, v4l2 = 0, kmsgrab = 0;
    AVCodecContext *codec_ctx = nullptr;
    AVCodec *codec = nullptr;
    AVFrame *frame = nullptr;
    AVPacket pkt;
    int lindex, opt;
    char *codec_name = nullptr;
    char *video_name = nullptr;
    char *pixel_format = nullptr, *size_window = nullptr;
    AVDictionary *opts = nullptr;
    AVDictionaryEntry *dict = nullptr;
    AVCodecParameters *codecpar;
    const AVInputFormat *ifmt = nullptr;
    int nframe = 1;
    int finished {};
    int i = 1;
    unsigned int a {};

    // --------- RGA
    int drm_rga_buf(int src_Width, int src_Height, int wStride, int hStride, int src_fd,
                    int src_format, int dst_Width, int dst_Height,
                    int dst_format, int frameSize, char *buf, char *viraddr);

    // --------- RKNN
    void initYOLOV5();
    bool selectLanguage();
    bool selectModel();
    bool defaultSettings();
    char* labelsListFile = (char*)"/usr/share/coco_80_labels_list.txt";
    char* model_path;
    char* model;
    int model_len = 0;

    // --------- OpenCV
    void save_and_display_crop_with_opencv(char* buffer, int width, int height);
    unsigned int hash_me(char *str);
};
