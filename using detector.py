from CarDetector import CarDetection

detector = CarDetection(capture_index='try.mp4', model_name=None, output='output.avi')

if __name__ == '__main':
    detector()
