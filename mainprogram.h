#ifndef MAINPROGRAM_H
#define MAINPROGRAM_H
#pragma once

#include <QMainWindow>
#include <QGraphicsDropShadowEffect>
#include <ffrknn.h>

QT_BEGIN_NAMESPACE
namespace Ui {
class MainProgram;
}
QT_END_NAMESPACE

class MainProgram : public QMainWindow
{
    Q_OBJECT

public:
    MainProgram(QWidget *parent = nullptr);
    ~MainProgram();

private:
    Ui::MainProgram *ui;
};
#endif // MAINPROGRAM_H
