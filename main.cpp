#include "mainprogram.h"
#include <QApplication>
#include <SDL2/SDL.h>
// undefine main otherwise SDL overrides it.
#undef main

int main(int argc, char *argv[])
{
    SDL_Init(SDL_INIT_EVERYTHING);
    QApplication a(argc, argv);
    MainProgram w;
    w.setWindowTitle("VonVision AI");
    w.show();
    return a.exec();
}
