#pragma once
#include <QWidget>
#include <QTimer>
#include <SDL2/SDL.h>

#include <drm_fourcc.h>
#include <xf86drm.h>
#include <xf86drmMode.h>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavdevice/avdevice.h>
#include <libavformat/avformat.h>
#include <libavutil/hwcontext_drm.h>
#include <libavutil/pixfmt.h>
}

#include <SDL_FontCache.h>
#include <postprocess.h>

#define FRAME_RATE 60
#define MS_PER_FRAME 1000 / FRAME_RATE

class SDLWidget : public QWidget
{
public :
    SDLWidget(QWidget* parent);
    virtual ~SDLWidget() override;

public slots:
    void SDLRepaint();

protected:
    SDL_Window* window {nullptr};
    SDL_Renderer* renderer {nullptr};
    SDL_Texture *texture {nullptr};
    FC_Font *font_small {nullptr};

private:
    virtual void Init() = 0;
    virtual void Update() = 0;
    virtual void OnResize(int w, int h) = 0;
    virtual int decode_and_display(AVCodecContext *dec_ctx, AVFrame *frame, AVPacket *pkt) = 0;

    void resizeEvent(QResizeEvent* event) override;
    void showEvent(QShowEvent*) override;

    QPaintEngine* paintEngine() const override;
    QTimer frameTimer;
};
