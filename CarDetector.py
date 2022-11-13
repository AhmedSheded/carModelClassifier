import torch
import numpy as np
import cv2
from time import time
import scipy.io
import os

if os.path.basename(os.getcwd()) != 'data':
    os.chdir('data')

prototxtFilePath = 'deploycompcars.prototxt'
modelFilePath = 'googlenet_finetune_web_car_iter_10000.caffemodel'
labels = scipy.io.loadmat('labels.mat')

mobilenet_model = 'mobilenet_iter_73000.caffemodel'
mobilnet_prottxt = 'deploy.prototxt'
net = cv2.dnn.readNetFromCaffe(prototxtFilePath,modelFilePath)
detect = cv2.dnn.readNetFromCaffe(mobilnet_prottxt, mobilenet_model)

class CarDetection:
    """
    Class implmentin yolov5 model with opencv
    """
    def __init__(self, capture_index, model_name, output=None):
        self.capture_index = capture_index
        self.model = self.load_model(model_name)
        self.classes = self.model.names
        self.width = int(cv2.VideoCapture(capture_index).get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cv2.VideoCapture(capture_index).get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.write = self.writer(output)
        # self.device = 'cuda' if self.cuda.is_available() else 'cpu'
        self.device = 'cpu'
        print('Using Device: ', self.device)

    def get_video_capture(self):
        return cv2.VideoCapture(self.capture_index)

    def writer(self, output):
        forcc = cv2.VideoWriter_fourcc('P','I','M','1')

        writer = cv2.VideoWriter(output, 30, forcc, (self.width, self.height))
        return writer

    def predict_frame(self, frame):
        if frame.shape[0]>2 and frame.shape[1]>2:
            meanValues=(104, 117, 123)
            imgWidth=224
            imgHeight=224
            blob = cv2.dnn.blobFromImage(frame, 1, (imgWidth, imgHeight), meanValues)
            net.setInput(blob)
            preds = net.forward()
            i = np.argmax(preds)
            campny = labels['make_model_names'][i][0][0]
            model = labels['make_model_names'][i][1][0]
        else:
            campny, model = '', ''
        return campny, model

    def load_model(self, model_name):
        if model_name:
            torch.hub.load('ultralytics/yolov5', 'custom', path=model_name, force_reload=True)
        else:
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        return model

    def score_frame(self, frame):
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord

    def class_to_label(self, x):
        """
        For a given label value, return corresponding string label.
        :param x: numeric label
        :return: corresponding string label
        """
        return self.classes[int(x)]

    def plot_boxes(self, results, frame):
        labels, cord = results
        cars = [i for i in labels if self.class_to_label(i)=='car']
        n = len(cars)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            # if self.class_to_label(i)=='car':
            row = cord[i]
            if row[4]>=0.2:

                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                campny, model = self.predict_frame(frame[x1:x2, y1:y2])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, 'Made: '+campny, (x1, y1-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame, 'Model: '+model, (x1, y1-2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return frame

    def __call__(self, *args, **kwargs):
        cap = self.get_video_capture()
        assert cap.isOpened()

        while True:
            ret, frame = cap.read()
            assert ret
            # frame = cv2.resize(frame, (416, 416))
            start_time = time()
            results = self.score_frame(frame)
            frame = self.plot_boxes(results, frame)

            endtime = time()
            fps = 1/np.round(endtime - start_time, 2)
            cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
            self.write.write(frame)
            cv2.imshow('Detection', frame)

            if cv2.waitKey(5) & 0xFF == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
