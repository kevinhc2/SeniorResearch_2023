import cv2
import numpy as np
from djitellopy import Tello
import time

with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

def detect_objects(img):
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(get_output_layers(net))
    return outs

def get_output_layers(net):
    return net.getUnconnectedOutLayersNames()

def draw_prediction(img, class_id, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])
    color = COLORS[class_id]
    cv2.rectangle(img, (x,y), (x_plus_w, y_plus_h), color, 2)
    cv2.circle(img, (x, y), 2, (255, 0, 0), thickness=2)
    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def display_preds(img, preds):
    class_ids = []
    confidences = []
    boxes = []
    c_thresh = 0.5

    classes_found = dict()
    focus_id = None
    global focus_mode, current_ooi

    for pred in preds: 
        for detection in pred:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > c_thresh:
                center_x = int(detection[0] * img.shape[0])
                center_y = int(detection[1] * img.shape[1])
                w = int(detection[2] * img.shape[0])
                h = int(detection[3] * img.shape[1])
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
                classes_found[class_id] = ([x, y, w, h], confidence)

    nms_thresh = 0.4
    indices = cv2.dnn.NMSBoxes(boxes, confidences, c_thresh, nms_thresh)
    detected_objects = []
    coordinates = []

    if focus_mode:
        focus_name = current_ooi[0]
        focus_id = classes.index(focus_name)
        if focus_id in classes_found:
            box = classes_found[focus_id][0]
            c_score = classes_found[focus_id][1]
            x, y, w, h = box
            x1 = round(x)
            y1 = round(y)
            x2 = round(x+w)
            y2 = round(y+h)
            draw_prediction(img, focus_id, x1, y1, x2, y2)
            detected_objects.append((focus_name, c_score))
            coordinates.append(((x1,y1),(x2,y2)))        
    else:
        for i in indices:
            try: box = boxes[i]
            except:
                i = i[0]
                box = boxes[i]  
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            x1 = round(x)
            y1 = round(y)
            x2 = round(x+w)
            y2 = round(y+h)
            draw_prediction(img, class_ids[i], x1, y1, x2, y2)
            detected_objects.append([classes[class_ids[i]], confidences[i]])
            coordinates.append(((x1,y1),(x2,y2)))

    return detected_objects, coordinates, img

def follow_object(tello, x, y, cur_width, cur_height):
    #for metrics if they are within certain number keep same
    # 1 is false
    #-1 is true
    width_s = abs(ooi_start_coors[1][0]-ooi_start_coors[0][0])
    default_x = ooi_start_coors[0][0]
    default_y = ooi_start_coors[0][1]
    closer = 0
    left = 0
    up = 0
    if(cur_width > 1.2 * width_s or cur_width < 0.8 * width_s):
        closer = 100 * ((cur_width - width_s) // frame_w)
    if(x < 0.8 * default_x or x > 1.2 * default_x):
        left = 100 * ((x - default_x) // frame_w)
    if(y > 1.2 * default_y or y < 0.8 * default_y):
        up = 100 * ((y - default_y) // frame_w)
    tello.send_rc_control(left, closer, up, 0)
    
def check_click(mouse_x, mouse_y, objs, coor_list):
    global focus_mode, current_ooi, ooi_start_coors
    if focus_mode:
        focus_mode = False
    else:
        for i in range(len(objs)):
            coors = coor_list[i]
            x0, y0 = coors[0]
            x1, y1 = coors[1]
            if (mouse_x > x0 and mouse_x < x1) and (mouse_y > y0 and mouse_y < y1):
                current_ooi = objs[i]
                ooi_start_coors = [(x0, y0), (x1, y1)]
                focus_mode = True
                break

def onMouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # print("x: " + str(x) + ", y: " + str(y))
        check_click(x, y, detected_objs, coors)
        
cv2.namedWindow('test', cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback('test', onMouse)
focus_mode = False
current_ooi = None
ooi_start_coors = None
frame_w = cv2.getWindowImageRect('test')[2]

tello = Tello()
tello.connect()
tello.streamon()
tello.takeoff()
tello.send_rc_control(0, 0, 15, 0)
time.sleep(4)
tello.send_rc_control(0, 0, 0, 0)
print(tello.get_battery())

while True:
    frame = tello.get_frame_read().frame
    outs = detect_objects(frame)
    detected_objs, coors, frame = display_preds(frame, outs)
    cv2.imshow('test', frame)
    if focus_mode and len(detected_objs) != 0:
        rect = coors[0]
        print(rect)
        x1, y1 = rect[0]
        x2, y2 = rect[1]
        follow_object(tello, x1, y1, abs(x2-x1), abs(y2-y1))
        
    key = cv2.waitKey(25)
    if key == 27: # escape key
        break
    tello.send_control_command('command')

cv2.destroyAllWindows()
tello.land()
tello.streamoff()