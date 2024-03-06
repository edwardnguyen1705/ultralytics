# My Utils

## Docker

Modifying dockerfile such that it contains only env not the source code. Then build a docker image.

```bash
$ t=yolo8:latest && docker build -f docker/Dockerfile_nosrc -t $t .
$ docker run -it --rm --gpus all --ipc=host \
    -v "$(pwd)":/usr/src/ultralytics \
    -v "$DS_DIR":/usr/datasets \
    $t

$ cd /usr/src/ultralytics
$ pip install --no-cache -e .
```

## Validation

```bash
# val and save prediction to a json file
$ python val.py
# val using the prediction json file
$ python val_json.py
```

## Validation on a custom dataset

1. Rename `YOLO` image and label files to int
2. Convert `YOLO` to `COCO` format.
2. Save predictions to a json file
3. Run `$ python val_json.py`
