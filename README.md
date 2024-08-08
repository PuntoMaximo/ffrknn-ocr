# VonVision AI

![alt text](https://viso.ai/wp-content/uploads/2021/03/yolo-v4-v5-wallpaper.jpg)

### Requisitos

* Placa Rockchip con NPU habilitada (rknn)
* SDL3 con aceleración de hardware para plataforma Rockchip
* FFmpeg con decoder rkmpp habilitado
* X11 con aceleración por hardware o Wayland/Weston con aceleración por hardware (mali)
* librga - Aceleración Rockchip 2D
* libyuv (para ultima version de FFmpeg)

## Modelo

![image](https://github.com/user-attachments/assets/b65146ce-1bce-4424-906b-bd1b525d7a03)

De base se usa un modelo pre-entrenado del algoritmo de deteccion YOLO (YOLOV5n) para detectar vehiculos y placas.
Se puede utilizar cualquier modelo de YOLOV5, pero debe asegurarse de convertir el modelo para su Rockchip usando RKNNToolkit.

## Dependencias

Cmake, QT5-QT6, SDL2, FFmpeg (con mp4,mkv,rtsp,rtmp,http support), librga, libyuv, librockchip_rkmpp

## Project Setup

* Una vez instaladas las dependencias... 
``` bash
cmake ./
```

* Compilar la app
```bash
make
```
