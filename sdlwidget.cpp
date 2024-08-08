#include <sdlwidget.h>
#include <iostream>

SDLWidget::SDLWidget(QWidget *parent) : QWidget(parent) {
    setAttribute(Qt::WA_OpaquePaintEvent);
    setAttribute(Qt::WA_PaintOnScreen);
    setAttribute(Qt::WA_NoSystemBackground);
    setFocusPolicy(Qt::StrongFocus);

    SDL_GL_SetAttribute(SDL_GL_RED_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
    SDL_GL_SetAttribute(SDL_GL_ACCELERATED_VISUAL, 1);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 2);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_ES);
}

void SDLWidget::resizeEvent(QResizeEvent* event) {
    int x = static_cast<int>(QWidget::width());
    int y = static_cast<int>(QWidget::height());
    OnResize(x, y);
}

void SDLWidget::showEvent(QShowEvent* event) {
    Init();
    connect(&frameTimer, &QTimer::timeout, this, &SDLWidget::SDLRepaint);
    frameTimer.setInterval(MS_PER_FRAME);
    frameTimer.start();
}

void SDLWidget::SDLRepaint() {
    // SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
    SDL_RenderClear(renderer);
    Update();
    // SDL_RenderPresent(renderer);
}

SDLWidget::~SDLWidget() {
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    exit(0);
}

QPaintEngine* SDLWidget::paintEngine() const {
    return nullptr;
}
