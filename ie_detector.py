from openvino.inference_engine import IECore
import cv2
import numpy as np

class InferenceEngineDetector:
    def __init__(self, configPath = None, weightsPath = None,
                 device = None, extension = None, classesPath = None):
        IEc = IECore()
        if (extension and device == 'CPU'):
            IEc.add_extension(extension, device)
        self.net = IEc.read_network(configPath, weightsPath)
        self.exec_net = IEc.load_network(self.net, device_name = device)
        with open(classesPath, 'r') as f:
            self.classes = [i.strip() for i in f]
    def _prepare_image(self, image, h, w):
        image = cv2.resize(image, (w, h))
        image = image.transpose((2, 0, 1))
        return image
    def detect(self, image):
        input_blob = next(iter(self.net.inputs))
        output_blob = next(iter(self.net.outputs))
        n, c, h, w = self.net.inputs[input_blob].shape
        image = self._prepare_image(image, h, w)
        output = self.exec_net.infer(inputs={input_blob: image})
        output = output[output_blob]
        return output
    def draw_detection(self, detections, image, confidence = 0.5, draw_text = True):
        detections = np.squeeze(detections)
        h, w, c = image.shape
        for classdet in detections:
            if (classdet[2] > confidence):
                image = cv2.rectangle(image, (int(classdet[3] * w), int(classdet[4] * h)),
                                    (int(classdet[5] * w), int(classdet[6] * h)),
                                    (0, 255, 0), 1)
                if (draw_text):
                    image = cv2.putText(image,
                                    self.classes[int(classdet[1])]
                                    + ' ' + str('{:.2f}'.format(classdet[2] * 100)) + '%',
                                    (int(classdet[3] * w - 5), int(classdet[4] * h - 5)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                    (0, 0, 255), 1)
        return image