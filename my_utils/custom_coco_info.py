

START_BOUNDING_BOX_ID = 1
# If necessary, pre-define category and its id
# PRE_DEFINE_CATEGORIES = {0: 'head', 1: 'face', 2: 'cellphone'}
PRE_DEFINE_CATEGORIES = {'head': 0, 'face': 1, 'cellphone': 2}

coco_info = {
    "year": "2024",
    "version": "1",
    "description": "X detection dataset",
    "contributor": "My org",
    "url": "",
    "date_created": "2024-03-04",
}

coco_licenses = [
    {
        "id": 1,
        "url": "",
        "name": "My org",
    }
]

jdict_init = {
    "info": coco_info,
    "licenses": coco_licenses,
    "images": [],
    "type": "instances",
    "annotations": [],
    "categories": [],
}
