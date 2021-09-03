import numpy as np
import cv2
import torch

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import colors, plot_one_box, colors
from utils.torch_utils import select_device


@torch.no_grad()
def detect(source, weights):

    classes = [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 56, 67]
    vehicles = [2, 5, 6, 7]
    animals = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    cycle = [1, 3]
    traffic_light = [9]
    stop_sign = [11]

    cap = cv2.VideoCapture(source)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(f'outputs/output0.mp4', fourcc, 15, (960, 540))
    while True:
        ret, im0s = cap.read()
        if not ret:
          break

        im0s = cv2.resize(im0s, (960, 540))
        device = select_device('')
        half = device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(960, s=stride)  # check img_size
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        if half:
            model.half()  # to FP16

        img = letterbox(im0s, 960, stride=stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        # Run inference4
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        # t0 = time.time()
        
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = model(img, augment='')[0]

        # Apply NMS
        pred = non_max_suppression(pred, 0.2, 0.6, classes=classes, agnostic='')

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            im0 = im0s.copy()

            # p = Path(p)  # to Path
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    if c in vehicles and conf > 0.5:
                        c = 2
                        label = f'vehicle'
                    elif c in animals:
                        c = 14
                        label = f'animal'
                    elif c in cycle  and conf > 0.5:
                        c = 1
                        label = 'cycle'
                    elif c == 0  and conf > 0.5:
                        label = 'person'
                    elif c in traffic_light and conf > 0.5:
                        light = im0[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
                        light = cv2.cvtColor(light, cv2.COLOR_BGR2RGB)
                        light = red_green_yellow(light)
                        if light != "":
                          label = f'light-{light}'
                        else:
                          continue
                    else:
                      continue
                    

                    plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=1)

        # Stream results
        out.write(im0)
        if cv2.waitKey(3) == ord('q'):
            out.release()
            break
    out.release()

def findNonZero(rgb_image):
    rows, cols, _ = rgb_image.shape
    counter = 0

    for row in range(rows):
      for col in range(cols):
        pixel = rgb_image[row, col]
        if sum(pixel) != 0:
          counter = counter + 1

    return counter

def red_green_yellow(rgb_image):
    '''Determines the Red, Green, and Yellow content in each image using HSV and
    experimentally determined thresholds. Returns a classification based on the
    values.
    '''
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    sum_saturation = np.sum(hsv[:,:,1]) # Sum the brightness values
    area = 32*32
    avg_saturation = sum_saturation / area # Find the average

    sat_low = int(avg_saturation * 1.3)
    val_low = 140

    lower_green = np.array([70,sat_low,val_low])
    upper_green = np.array([100,255,255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    green_result = cv2.bitwise_and(rgb_image, rgb_image, mask = green_mask)

    # Yellow
    lower_yellow = np.array([10,sat_low,val_low])
    upper_yellow = np.array([60,255,255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    yellow_result = cv2.bitwise_and(rgb_image, rgb_image, mask = yellow_mask)

    # Red
    lower_red = np.array([150,sat_low,val_low])
    upper_red = np.array([180,255,255])
    red_mask = cv2.inRange(hsv, lower_red, upper_red)
    red_result = cv2.bitwise_and(rgb_image, rgb_image, mask = red_mask)

    sum_green = findNonZero(green_result)
    sum_yellow = findNonZero(yellow_result)
    sum_red = findNonZero(red_result)

    print('red' + str(sum_red))
    print('yellow' + str(sum_yellow))
    print('green' + str(sum_green))

    if sum_red >= sum_yellow and sum_red >= sum_green and sum_red >= 3:
      return "STOP"
    elif sum_yellow >= sum_green and sum_yellow >= 3:
      return "STOP"
    elif sum_green >= 5:
      return "GO"
    else:
      return ""

if __name__ == '__main__':
    detect('test/1.mp4', 'weights/yolov5s.pt')
