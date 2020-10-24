import ie_classifier as ic
import argparse
import logging as log
import sys
import cv2

def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', help='Path to an .xml \
     file with a trained model.', required=True, type=str)
    parser.add_argument('-w', '--weights', help='Path to an .bin file \
     with a trained weights.', required=True, type=str)
    parser.add_argument('-i', '--input', help='Path to \
     image file', required=True, type=str)
    parser.add_argument('-c', '--classes', help='File containing \
     classnames', type=str, default=None)
    parser.add_argument('-d', '--device', help='Device name',
                        default = "CPU", type = str)
    parser.add_argument('-e', '--cpu_extension', help='For custom',
                        default = None, type = str)
    return parser

def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s",
    level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()
    log.info("Start IE classification sample")
    ie_classifier = ic.InferenceEngineClassifier(configPath=args.model,
    weightsPath=args.weights, device=args.device, extension=args.cpu_extension,
    classesPath=args.classes)
    img = cv2.imread(args.input)
    prob = ie_classifier.classify(img)
    predictions = ie_classifier.get_top(prob, 5)
    log.info("Predictions: " + str(predictions))
    return

if __name__ == '__main__':
 sys.exit(main())