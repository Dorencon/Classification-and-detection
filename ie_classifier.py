from openvino.inference_engine import IECore
import cv2
import numpy as np

class InferenceEngineClassifier:
    def __init__(self, configPath = None, weightsPath = None, device = None, extension = None, classesPath = None):
        IEc = IECore()
        if (extension and device == "CPU"):
            IEc.add_extension(extension, device)
        self.net = IEc.read_network(configPath, weightsPath)
        self.exec_net = IEc.load_network(self.net, device_name=device)
        with open(classesPath, 'r') as f:
            self.classes = [i.strip() for i in f]
    def _prepare_image(self, image, h, w):
        image = cv2.resize(image, (w, h))
        image = image.transpose((2, 0, 1))
        return image
    def classify(self, image):
        input_blob = next(iter(self.net.inputs))
        out_blob = next(iter(self.net.outputs))
        n, c, h, w = self.net.inputs[input_blob].shape
        image = self._prepare_image(image, h, w)
        output = self.exec_net.infer(inputs={input_blob: image})
        output = output[out_blob]
        return output
    def get_top(self, prob, topN = 1):
        prob = np.squeeze(prob)
        top = np.argsort(prob)
        out = []
        for i in top[1000 - topN:1000]:
            out.append([self.classes[i], '{:.15f}'.format(prob[i])])
        out.reverse()
        return out