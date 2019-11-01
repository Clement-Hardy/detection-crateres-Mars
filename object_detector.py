import numpy as np
import cv2
from imgaug import augmenters as aug
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib



class ObjectDetector:
    
    def __init__(self, height=256, width=256, epoch=8):
        self.epoch = epoch
        self.height = height
        self.width = width
        
        
        config = Detector_Config()
        
        self.model = modellib.MaskRCNN(mode='training', config=config, model_dir='.')
        
    def fit(self, X, y):
        valid_ratio = 0.1
        self.valid_ratio = valid_ratio
        indice_max = np.floor(len(X) * (1 - valid_ratio)).astype(int)
        self.dataset_train = DetectorDataset(image=X[0:indice_max],
                                             label=y[0:indice_max],
                                             height=self.height,
                                             width=self.width)
        self.dataset_train.prepare()
        
        self.dataset_validation = DetectorDataset(image=X[indice_max:len(X)],
                                                  label=y[indice_max:len(X)],
                                                  height=self.height,
                                                  width=self.width)
        self.dataset_validation.prepare()
        
        augmentation = self.augmentation_data()
        
        
        LEARNING_RATE = 0.01
        
        
        COCO_PATH = "mask_rcnn_coco.h5"
        utils.download_trained_weights(COCO_PATH)
        self.model.load_weights(COCO_PATH, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
        
        self.model.train(self.dataset_train,
                    self.dataset_validation, 
                    learning_rate=LEARNING_RATE/5., 
                    epochs=self.epoch, 
                    layers='all',
                    augmentation=augmentation)

    def rectangle_to_circle(self, pred, resize_original_shape=True, height=256):
        circle = []
        for i in range(len(pred['rois'])):
            x_min,y_min, x_max,y_max = pred["rois"][i]
            r = (((x_max - x_min) + (y_max - y_min)) / 2.) / 2.
            x = x_min + (x_max - x_min) / 2.
            y = y_min + (y_max - y_min) / 2.
            label = (x, y, r)
            if resize_original_shape:
                label = tuple(np.insert(np.multiply(label, height/self.height), 0, pred['scores'][i]))
            circle.append(label)
        return circle

    def predict(self, X):
        inference_config = InferenceConfig()
        
        self.model_pred = modellib.MaskRCNN(mode='inference', 
                          config=inference_config,
                          model_dir='.')

        self.model_pred.load_weights(self.model.find_last(), by_name=True)
        
        pred = []
        for i in range(len(X)):
            image_resize = cv2.resize(X[i], (self.height, self.width))
            image = np.stack((image_resize,) * 3, -1)
            pred_image = self.model_pred.detect([image])[0]
            pred.append(self.rectangle_to_circle(pred=pred_image, height=X[0].shape[0], resize_original_shape=True))
        
        pred_array = np.empty(len(pred), dtype=object)
        pred_array[:] = pred
        return pred_array
    
    
    def augmentation_data(self):
        augmentation = aug.Sequential([
            aug.OneOf([
                aug.Affine(rotate=0),
                aug.Affine(rotate=90),
                aug.Affine(rotate=180),
                aug.Affine(rotate=270),
            ]),
            aug.Fliplr(0.5),
            aug.Flipud(0.5),
            aug.ContrastNormalization((0.5, 1.5)),
        ])
        return augmentation

class DetectorDataset(utils.Dataset):
    """Dataset class for training our dataset.
    """

    def __init__(self, image, label, height, width):
        super().__init__(self)
        

        self.add_class('crater', 1, 'Crater')
        self.X_train = image
        self.initial_height = image[0].shape[0]
        self.initial_width = image[0].shape[1]
        
        for i in range(len(image)):
            self.add_image('crater', image_id=i, label=label[i], height=height, width=width, path=i)
     
    def circle_to_rectangle(self, label, height, width):
        x = np.array([label[0]-label[2], label[0]+label[2]])
        x[x<0] = 0
        x[x>=width] = width - 1
    
        y = np.array([label[1]-label[2], label[1]+label[2]])
        y[y<0] = 0
        y[y>=height] = height - 1
    
        return np.concatenate((x, y)).astype(int)
            
        
    def resize_label(self, label, height): 
        return np.multiply(label, (height/self.initial_height))
    
    def rle_decode(self, label, shape=(224, 224)):
        mask = np.zeros(shape)
        mask[label[0]:(label[1] + 1), label[2]:(label[3] + 1)] = 1
        return mask 

      
    def load_image(self, image_id):
        
        info = self.image_info[image_id]
        image = cv2.resize(self.X_train[image_id], (info['height'], info['width']))
        image = np.stack((image,) * 3, -1)
        
        return image
    
    def load_mask(self, image_id):
        info = self.image_info[image_id]
        
        
        labels = self.resize_label(height=info['height'],
                            label=info['label'])
        count = len(labels)
        if count == 0:
            mask = np.zeros((info['height'], info['width'], 1), dtype=np.uint8)
            class_ids = np.zeros((1,), dtype=np.int32)
        else:
            mask = np.zeros((info['height'], info['width'], count), dtype=np.uint8)
            class_ids = np.zeros((count,), dtype=np.int32)
            for i in range(len(labels)):
                label = self.circle_to_rectangle(height=info['height'],width=info['width'],label=labels[i])
                mask[:, :, i] = self.rle_decode(label=label, shape=(info['height'], info['width']))
                class_ids[i] = 1
        return mask.astype(np.bool), class_ids.astype(np.int32)

class Detector_Config(Config):    
    NAME = 'mars'
    
    GPU_COUNT = 1
    
    BACKBONE = 'resnet50'
    
    NUM_CLASSES = 2  
    IMAGES_PER_GPU = 8
    STEPS_PER_EPOCH = 1125
    
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256
    RPN_ANCHOR_SCALES = (5, 8, 10, 15, 22)
    TRAIN_ROIS_PER_IMAGE = 64
    MAX_GT_INSTANCES = 14
    DETECTION_MAX_INSTANCES = 10
    DETECTION_MIN_CONFIDENCE = 0.95
    DETECTION_NMS_THRESHOLD = 0.0

    VALIDATION_STEPS = 100
    

class InferenceConfig(Detector_Config):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1