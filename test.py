from operator import index
import torch, torchvision
print(torch.__version__, torch.cuda.is_available())
assert torch.__version__.startswith("1.10")   # please manually install torch 1.9 if Colab changes its default version
# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random


# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectron2.data.datasets import register_coco_instances
register_coco_instances("test", {}, "anh/test_json/data.json", "anh/test")
sample_metadata_test = MetadataCatalog.get("test")
dataset_dicts_test = DatasetCatalog.get("test")


from detectron2.config import get_cfg
import os

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TEST = ("test",)   # no metrics implemented for this dataset
cfg.DATALOADER.NUM_WORKERS = 12
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9   # set the testing threshold for this model
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 1024 # faster, and good enough for this toy dataset
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # 3 classes (Person, Helmet, Car)

predictor = DefaultPredictor(cfg)

from detectron2.utils.visualizer import ColorMode
indexs = 1
for d in (dataset_dicts_test):    
    im = cv2.imread(d["file_name"])
    #cv2.imwrite(f'image/{indexs}.jpg',im)
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=sample_metadata_test, 
                   scale=0.9, 
                   instance_mode=ColorMode.SEGMENTATION   # remove the colors of unsegmented pixels
    )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"),scales=9.0)
    cv2.imwrite(f'image_pre/{indexs}.jpg',v.get_image()[:, :, ::-1])
    mask = np.zeros(im.shape, dtype=np.uint8)
    #outputs = predictor(mask)
    m = Visualizer(mask[:, :, ::-1],
                   metadata=sample_metadata_test, 
                   scale=0.9, 
                   instance_mode=ColorMode.SEGMENTATION   # remove the colors of unsegmented pixels
    )
    m = m.draw_instance_predictions(outputs["instances"].to("cpu"),scales=10.5)
    cv2.imwrite(f'image_mask/{indexs}.jpg',m.get_image()[:, :, ::-1])
    indexs +=1
    # cv2.imshow("img",v.get_image()[:, :, ::-1])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()