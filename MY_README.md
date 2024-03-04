# Notes

## Docker

Modifying dockerfile such that it contains only env not the source code. Then build a docker image.

```bash
$ t=yolo8:latest && docker build -f docker/Dockerfile_nosrc -t $t .
$ docker run -it --rm --gpus all -v "$(pwd)":/usr/src/ultralytics $t

$ cd /usr/src/ultralytics
$ pip install --no-cache -e ".[export]"
```
