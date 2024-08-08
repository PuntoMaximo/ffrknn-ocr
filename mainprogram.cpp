#include "mainprogram.h"
#include "./ui_mainprogram.h"

MainProgram::MainProgram(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainProgram)
{
    ui->setupUi(this);
}

MainProgram::~MainProgram()
{
    delete ui;
}
