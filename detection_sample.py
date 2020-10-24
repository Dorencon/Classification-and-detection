import ie_detector as id
import logging as log
import cv2
import argparse
import sys

def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', help = 'Path to an .xml \
    file with a trained model.', required = True, type = str)
    parser.add_argument('-w', '--weights', help = 'Path to an .bin file \
    with a trained weights.', required = True, type = str)
    parser.add_argument('-i', '--input', help = 'Path to \
    image file.', required = True, type = str)
    parser.add_argument('-d', '--device', help = 'Device name',
                        default='CPU', type = str)
    parser.add_argument('-l', '--cpu_extension',
                        help = 'MKLDNN (CPU)-targeted custom layers. \
                        Absolute path to a shared library with the kernels implementation',
                        type = str, default=None)
    parser.add_argument('-c', '--classes', help = 'File containing \
    classnames', type = str, default=None)
    return parser

def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s",
                    level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()
    log.info("Start IE detection sample")
    ie_detector = id.InferenceEngineDetector(configPath=args.model,
                                             weightsPath=args.weights,
                                             device=args.device,
                                             extension=args.cpu_extension,
                                             classesPath=args.classes)
    img = cv2.imread(args.input)
    detections = ie_detector.detect(img)
    image_detected = ie_detector.draw_detection(detections, img)
    cv2.imshow('Image with detections', image_detected)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return

if (__name__  == '__main__'):
    sys.exit(main())