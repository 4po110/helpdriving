import numpy as np
import cv2
import torch
import platform
import yaml
from pathlib import Path


from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import colors, plot_one_box, colors
from utils.torch_utils import select_device

import tensorflow as tf
from tensorflow import keras

EDGETPU_SHARED_LIB = {
  'Linux': 'libedgetpu.so.1',
  'Darwin': 'libedgetpu.1.dylib',
  'Windows': 'edgetpu.dll'
}[platform.system()]

@torch.no_grad()
def detect(source, weights):

    classes = [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 56, 67]
    vehicles = [2, 5, 6, 7]
    animals = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    cycle = [1, 3]
    traffic_light = [9]
    stop_sign = [11]

    device = select_device('')

    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(
        model_path=weights,
        experimental_delegates=
            [tf.lite.experimental.load_delegate(EDGETPU_SHARED_LIB)])
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Load model
    with open('data/coco.yaml') as f:
        names = yaml.load(f, Loader=yaml.FullLoader)['names']  # class names (assume COCO)
    # if half:
    #     model.half()  # to FP16

    # Run inference4
    # if device.type != 'cpu':
    #     model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    
    imgsz = [320, 320]

    cap = cv2.VideoCapture(source)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(f'outputs/output0.mp4', fourcc, 15, (960, 540))
    while True:
        ret, im0s = cap.read()
        if not ret:
          break

        im0s = cv2.resize(im0s, (960, 540))

        img = letterbox(im0s, [320, 320], stride=None, auto=False)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        # t0 = time.time()
        
        img = torch.from_numpy(img).to(device)
        img.float()  # uint8 to fp16/32
        img = img / 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        input_data = img.permute(0, 2, 3, 1).cpu().numpy()
        scale, zero_point = input_details[0]['quantization']
        input_data = input_data / scale + zero_point
        input_data = input_data.astype(np.uint8)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        # output_data = interpreter.get_tensor(output_details[0]['index'])
        # pred = torch.tensor(output_data)

        yaml_file = Path('./models/yolov5s.yaml').name
        with open('./models/yolov5s.yaml') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)

        anchors = cfg['anchors']
        nc = cfg['nc']
        nl = len(anchors)
        x = [torch.tensor(interpreter.get_tensor(output_details[i]['index']), device=device) for i in range(nl)]
        for i in range(nl):
            scale, zero_point = output_details[i]['quantization']
            x[i] = x[i].float()
            x[i] = (x[i] - zero_point) * scale

        def _make_grid(nx=20, ny=20):
            yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
            return torch.stack((xv, yv), 2).view((1, 1, ny * nx, 2)).float()

        no = nc + 5
        grid = [torch.zeros(1)] * nl  # init grid
        a = torch.tensor(anchors).float().view(nl, -1, 2).to(device)
        anchor_grid = a.clone().view(nl, 1, -1, 1, 2)  # shape(nl,1,na,1,2)
        z = []  # inference output
        for i in range(nl):
            _, _, ny_nx, _ = x[i].shape
            r = imgsz[0] / imgsz[1]
            nx = int(np.sqrt(ny_nx / r))
            ny = int(r * nx)
            grid[i] = _make_grid(nx, ny).to(x[i].device)
            stride = imgsz[0] // ny
            y = x[i].sigmoid()
            y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + grid[i].to(x[i].device)) * stride  # xy
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor_grid[i]  # wh
            z.append(y.view(-1, no))

        pred = torch.unsqueeze(torch.cat(z, 0), 0)

        # Inference
        # pred = model(img, augment='')[0]

        # Apply NMS
        pred = non_max_suppression(pred, 0.1, 0.6, classes=classes, agnostic='')

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
                    xmin = int(xyxy[0])
                    ymin = int(xyxy[1])
                    xmax = int(xyxy[2])
                    ymax = int(xyxy[3])
                    if c in vehicles and conf > 0.4:
                        c = 2
                        label = f'vehicle'
                        if 480 > xmin and 480 < xmax and 270 > ymin and 270 < ymax:
                            plot_one_box(xyxy, im0, label=label, color=(0, 0, 255), line_thickness=1)
                    elif c in animals:
                        c = 14
                        label = f'animal'
                        plot_one_box(xyxy, im0, label=label, color=(0, 0, 255), line_thickness=1)
                    elif c in cycle  and conf > 0.4:
                        c = 1
                        label = 'cycle'
                    elif c == 0  and conf > 0.4:
                        label = 'person'
                    elif c in traffic_light and conf > 0.4:
                        light = im0[ymin:ymax, xmin:xmax]
                        print(im0.shape)
                        light = cv2.cvtColor(light, cv2.COLOR_BGR2RGB)
                        light = red_green_yellow(light)
                        if light == "STOP":
                          label = f'{light}'
                          plot_one_box(xyxy, im0, label=label, color=(0, 0, 255), line_thickness=1)
                        else:
                          continue
                    else:
                      continue
                    

                    # plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=1)

        # Stream results
        cv2.imshow('', im0)
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
    detect('test/1.mp4', 'weights/yolov5s-int8_edgetpu.tflite')
