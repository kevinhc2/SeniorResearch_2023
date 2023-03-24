import cv2
import argparse
import numpy as np
import time

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help = 'path to input image')
ap.add_argument('-c', '--config', required=True, help = 'path to yolo config file')
ap.add_argument('-w', '--weights', required=True, help = 'path to yolo pre-trained weights')
ap.add_argument('-cl', '--classes', required=True, help = 'path to text file containing class names')
args = ap.parse_args()

def get_output_layers(net):
    layer_names = net.getLayerNames()
    try: output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    except: output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])
    color = COLORS[class_id]
    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

image = cv2.imread(args.image)
Width = image.shape[1]
Height = image.shape[0]
scale = 0.00392
classes = None

with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
net = cv2.dnn.readNet(args.weights, args.config)
blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)
net.setInput(blob)
outs = net.forward(get_output_layers(net))

class_ids = []
confidences = []
boxes = []
conf_threshold = 0.5
nms_threshold = 0.4

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0] * Width)
            center_y = int(detection[1] * Height)
            w = int(detection[2] * Width)
            h = int(detection[3] * Height)
            x = center_x - w / 2
            y = center_y - h / 2
            class_ids.append(class_id)
            confidences.append(float(confidence))
            boxes.append([x, y, w, h])

indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
detected_objects = []
coordinates = []

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
    draw_prediction(image, class_ids[i], confidences[i], x1, y1, x2, y2)
    detected_objects.append([classes[class_ids[i]], confidences[i]])
    coordinates.append(((x1,y1),(x2,y2)))

# print("hi")
print("Objects detected with coordinates:")
for i in range(0,len(detected_objects)):
    print("Object " + str(i+1) + ":")
    print("Classification:",detected_objects[i][0])
    print("Confidence:",detected_objects[i][1])
    print("Top-Left Coordinate:",coordinates[i][0])
    print("Bottom-Right Coordinate:",coordinates[i][1])
    print()
    time.sleep(1)


print("Choosing an object...")
choice = int(input("Which object would you like to select: "))
while(choice > len(detected_objects) or choice <= 0):
    print("Please insert a number from the range", 1, "to", len(detected_objects),"\n")
    choice = int(input("Which object would you like to select: "))
print("You have selected Object", choice, "classified as a", detected_objects[choice-1][0])

default_pos = coordinates[choice-1]
width_s = coordinates[choice-1][0][0] - coordinates[choice-1][1][0]
height_s = coordinates[choice-1][1][1] - coordinates[choice-1][0][1]
coords_by_frame = [default_pos]
size_by_frame = [[width_s, height_s]]

def follow_object(tello, x, y, cur_width, cur_height):
    #for metrics if they are within certain number keep same
    #1 is false
    #-1 is true
    coords_by_frame.append([x, y])
    size_by_frame.append([cur_width, cur_height])
    closer = 0
    left = 0
    up = 0
    if(cur_width > 1.2 * width_s or cur_width < 0.8 * width_s):
        closer = 100 * ((cur_width - width_s) / Width)
    if(x < 0.8 * default_pos[0][0] or x > 1.2 * default_pos[0][0]):
        left = 100 * ((x - default_pos[0][0]) / Width)
    if(y > 1.2 * default_pos[0][1] or y < 0.8 * default_pos[0][1]):
        up = 100 * ((y - default_pos[0][1]) / Width)
    tello.send_rc_control(left, closer, up, 0)
     
    


# cv2.imshow("object detection", image)
# cv2.waitKey()
    
# cv2.imwrite("object-detection.jpg", image)
# cv2.destroyAllWindows()