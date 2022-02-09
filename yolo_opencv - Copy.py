from io import StringIO

import cv2
import argparse
import numpy as np
import streamlit as st

# ap = argparse.ArgumentParser()
# ap.add_argument('-i', '--image', required=True,
#                 help='path to input image')
# args = ap.parse_args()

cam = cv2.VideoCapture(0)

cv2.namedWindow("test")


def take_picture(count):
    img_counter = 0
    img_names = []
    while img_counter < 1:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        cv2.imshow("test", frame)

        k = cv2.waitKey(1)
        if k % 256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            cam.release()
            cv2.destroyAllWindows()
            break
        elif k % 256 == 32:
            # SPACE pressed
            img_name = "opencv_frame_{}.png".format(count)
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            img_names.append(img_name)
            img_counter += 1

    return img_names

def get_output_layers(net):
    layer_names = net.getLayerNames()

    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])

    # st.subheader(label)
    print(label)
    color = COLORS[class_id]

    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)

    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


# st.title("Object Detection System: ")
# uploaded_img = st.sidebar.file_uploader("Choose a image File")

# image0 = uploaded_img.read()
# st.sidebar.subheader("Input Image: ")
# st.sidebar.image(image0)
# image = cv2.imread(uploaded_img.name)

scale = 0.00392
classes = None

with open('yolov3.txt', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

count = 0

while count<2:
    img_names = take_picture(count)
    image = cv2.imread(img_names[0])

    Width = image.shape[1]
    Height = image.shape[0]
    blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)

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
    # st.subheader("Objects Contained within the input Image: ")
    for i in indices:
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))
    cv2.imwrite("object-detection-{}.jpg".format(count), image)
    count+=1

# image  = cv2.resize(image, (500, 500))
# cv2.imshow("object detection", image)
# cv2.waitKey()
# st.write("\n")
# st.subheader("Output Image: ")
# cv2.imwrite("object-detection.jpg", image)
# st.image("object-detection.jpg")
cam.release()
cv2.destroyAllWindows()

