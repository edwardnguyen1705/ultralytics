
def to_coco_box(yolo_box, img_h, img_w):
    xc, yc, w, h = yolo_box
    xmin = float((xc * img_w) - (w * img_w) / 2.0)
    ymin = float((yc * img_h) - (h * img_h) / 2.0)
    w = float(w * img_w)
    h = float(h * img_h)
    return [xmin, ymin, w, h]
    
def xyxy2xywh(xyxy):
    xmin, ymin, xmax, ymax = xyxy
    w = float((xmax - xmin))
    h = float((ymax - ymin))
    return [xmin, ymin, w, h]
    
def valid_coco_box(box, img_h, img_w):
    xmin, ymin, w, h = box
    xmax, ymax = xmin + w, ymin + h
    xmin = max(xmin, 0)
    ymin = max(ymin, 0)
    xmax = min(xmax, img_w)
    ymax = min(ymax, img_h)
    if (xmin > xmax) or (ymin > ymax):
        return []
    return xyxy2xywh([xmin, ymin, xmax, ymax])
