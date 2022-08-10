from urllib import error
import cv2
import argparse
import threading
import torch
import math
import numpy as np

from yolov6.layers.common import DetectBackend
from yolov6.data.datasets import LoadData
from yolov6.data.data_augment import letterbox
from yolov6.utils.nms import non_max_suppression

RTSP_URL = 'rtmp://192.168.1.80/bcs/channel0_main.bcs?channel=0&stream=0&user=aipsystem&password=Abc12345'

class Detector:
    def __init__(self, weights:str, device:str, conf_thres:float, iou_thres:float):

        self.__dict__.update(locals())

        # Init model
        self.device = device
        self.img_size = [1280,1280]
        cuda = self.device != 'cpu' and torch.cuda.is_available()
        self.device = torch.device('cuda:0' if cuda else 'cpu')
        self.model = DetectBackend(weights, device=self.device)
        self.stride = self.model.stride
        self.class_names = ['fire', 'smoke']
        self.img_size = self.check_img_size(self.img_size, s=self.stride)  # check image size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        self.model.model.float()
        # half = False

        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, *self.img_size).to(self.device).type_as(next(self.model.model.parameters())))  # warmup

        # Switch model to deploy status
        self.model_switch(self.model, self.img_size)
        # print(self.model)

    def model_switch(self, model, img_size):
        ''' Model switch to deploy status '''
        from yolov6.layers.common import RepVGGBlock
        for layer in model.modules():
            if isinstance(layer, RepVGGBlock):
                layer.switch_to_deploy()

    def check_img_size(self, img_size, s=32, floor=0):
        """Make sure image size is a multiple of stride s in each dimension, and return a new shape list of image."""
        if isinstance(img_size, int):  # integer i.e. img_size=640
            new_size = max(self.make_divisible(img_size, int(s)), floor)
        elif isinstance(img_size, list):  # list i.e. img_size=[640, 480]
            new_size = [max(self.make_divisible(x, int(s)), floor) for x in img_size]
        else:
            raise Exception(f"Unsupported type of img_size: {type(img_size)}")

        if new_size != img_size:
            print(f'WARNING: --img-size {img_size} must be multiple of max stride {s}, updating to {new_size}')
        return new_size if isinstance(img_size,list) else [new_size]*2

    def make_divisible(self, x, divisor):
        # Upward revision the value x to make it evenly divisible by the divisor.
        return math.ceil(x / divisor) * divisor

    def detect_fire(self, img):
        _img, img = Detector.precess_image(img, self.img_size, self.stride)
        if len(_img.shape) == 3:
            _img = _img[None]
        pred_results = self.model(_img)
        det = non_max_suppression(pred_results, self.conf_thres, self.iou_thres, self.class_names, agnostic=False, max_det=1000)[0]
        det[:, :4] = Detector.rescale(_img.shape[2:], det[:, :4], _img.shape).round()
        for *xyxy, conf, cls in reversed(det):
            print(xyxy, conf, cls)

    def precess_image(img_src, img_size, stride):
        '''Process image before image inference.'''
        image = letterbox(img_src, img_size, stride=stride)[0]
        # Convert
        image = image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        image = torch.from_numpy(np.ascontiguousarray(image))
        image = image.float()
        image /= 255  # 0 - 255 to 0.0 - 1.0

        return image, img_src

    def rescale(ori_shape, boxes, target_shape):
        '''Rescale the output to the original image shape'''
        ratio = min(ori_shape[0] / target_shape[0], ori_shape[1] / target_shape[1])
        padding = (ori_shape[1] - target_shape[1] * ratio) / 2, (ori_shape[0] - target_shape[0] * ratio) / 2

        boxes[:, [0, 2]] -= padding[0]
        boxes[:, [1, 3]] -= padding[1]
        boxes[:, :4] /= ratio

        boxes[:, 0].clamp_(0, target_shape[1])  # x1
        boxes[:, 1].clamp_(0, target_shape[0])  # y1
        boxes[:, 2].clamp_(0, target_shape[1])  # x2
        boxes[:, 3].clamp_(0, target_shape[0])  # y2

        return boxes

def catch_frames():
    global ret, frame
    ret, frame = cap.read()
    first_frame_caught.set()
    while True:
        ret, frame = cap.read()
        if stop_program:
            break
    cap.release()
    cv2.destroyAllWindows()
    # print('Stopped reading frames')

def show_frames(detector:Detector, window_name:str, hide_labels:bool, hide_conf:bool):
    global stop_program
    first_frame_caught.wait()
    while True:
        output_frame = cv2.resize(frame, (int(frame_width/2), int(frame_height/2)))
        if ret:
            cv2.imshow(window_name, output_frame)
            detector.detect_fire(frame)
        else:
            break

        if cv2.waitKey(1) == 27 or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break
    stop_program = True

def main(weights, source, window_name, conf_thres, iou_thres, device, hide_labels, hide_conf):
    global cap, frame_width, frame_height, detector

    detector = Detector(weights,device,conf_thres,iou_thres)

    cap = cv2.VideoCapture(source)
    frame_width, frame_height = cap.get(3), cap.get(4)

    global first_frame_caught, stop_program
    first_frame_caught = threading.Event()
    stop_program = False
    threading.Thread(target=catch_frames, args=()).start()
    threading.Thread(target=show_frames, args=(detector,window_name,hide_labels,hide_conf,)).start()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AIP fire detection system', add_help=True)
    parser.add_argument('--weights', type=str, default='weights/yolov6n_fire.pt', help='model path(s) for inference.')
    parser.add_argument('--source', type=str, default=RTSP_URL, help='the source camera')
    parser.add_argument('--window-name', type=str, default='AIP camera', help='the name of the window')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='confidence threshold for inference.')
    parser.add_argument('--iou-thres', type=float, default=0.1, help='NMS IoU threshold for inference.')
    parser.add_argument('--device', default='cpu', help='device to run our model i.e. 0 or 0,1,2,3 or cpu.')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels.')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences.')
    args = parser.parse_args()
    main(**vars(args))