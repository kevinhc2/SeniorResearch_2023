
import cv2
import numpy as np
from djitellopy import Tello
import time

with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

net = cv2.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')

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
    cv2.putText(img, label, (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

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

def follow_object(tello, x, cur_width, cur_height, prev_err):
    pid = [0.4, 0.4]
    error = x-416//2 #how far away is the face from the center of the screen?

    #complicated error stuff
    speed = pid[0]*error + pid[1]*(error-prev_err)
    speed = int(np.clip(speed, -100, 100))

    prev_width = abs(ooi_start_coors[0][0]-ooi_start_coors[1][0])
    prev_height = abs(ooi_start_coors[0][1]-ooi_start_coors[1][1])
    start_area = prev_height*prev_width

    rectArea = cur_width*cur_height #get area of rect 
    if rectArea > start_area*1.1: #if too close, move back 
        moveDist = -20
    elif rectArea < start_area*0.9 and rectArea != 0: #if too far, move foward
        moveDist = 20
    else: #within acceptable range
        moveDist = 0
    
    if x == 0:
        speed = 0
        error = 0

    tello.send_rc_control(0, moveDist, 0, speed)
    return error 
    
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
        
cv2.namedWindow('dronecam', cv2.WINDOW_AUTOSIZE)
cv2.resizeWindow('dronecam', 416, 416) 
cv2.setMouseCallback('dronecam', onMouse)
focus_mode = False
current_ooi = None
ooi_start_coors = None
frame_w = cv2.getWindowImageRect('dronecam')[2]
frame_h = cv2.getWindowImageRect('dronecam')[3]

tello = Tello()
tello.connect()
tello.streamon()
tello.takeoff()
# tello.send_rc_control(0, 0, 5, 0)
# time.sleep(4)
# tello.send_rc_control(0, 0, 0, 0)
print(tello.get_battery())
frames_rec = 0
prevErr = 0

fourcc = cv2.VideoWriter_fourcc(*'XVID')
vidOutput = cv2.VideoWriter("demovideo.avi", fourcc, 4.0, (416, 416))

while True:
    if frames_rec == 0:
        start = time.time()
    frame = cv2.resize(tello.get_frame_read().frame, (416, 416))
    frames_rec += 1
    outs = detect_objects(frame)
    detected_objs, coors, frame = display_preds(frame, outs)
    if frames_rec >= 4:
        fps = frames_rec/(time.time()-start)
        frame = cv2.putText(frame, "FPS: " + str(round(fps, 1)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow('dronecam', frame)
    vidOutput.write(frame)
    if focus_mode and len(detected_objs) != 0:
        rect = coors[0]
        print(rect)
        x1, y1 = rect[0]
        x2, y2 = rect[1]
        prevErr = follow_object(tello, (x1+x2)//2, abs(x2-x1), abs(y2-y1), prevErr)
    
    key = cv2.waitKey(25)
    if key == 27: # escape key
        break
    tello.send_control_command('command')

cv2.destroyAllWindows()
tello.land()
tello.streamoff()
vidOutput.release()