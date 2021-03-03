import os
import glob
import json
import cv2
import numpy as np
import time

net = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel"
)

result_path = './saved/result_img/'
video_path = './youtube_data/video/test.mp4'


def main():
    configs = json.load(open("./configs/fer2013_config.json"))
    image_size = (configs["image_size"], configs["image_size"])

    
    vid = cv2.VideoCapture(video_path)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (640,360))

    rate = round(vid.get(5))
    print(type(rate))
    FrameNumber = vid.get(7)
    duration = FrameNumber / rate
    print("The duration is %f s" % duration)
    print("The number of frame is %d " % FrameNumber)
    print("The rate of video is %d " % rate)

    start_time = time.time()
    while True:
        ret, frame = vid.read()
        if frame is None or ret is not True:
            break
        
        frame = np.fliplr(frame).astype(np.uint8)
        h, w = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0),
        )
        net.setInput(blob)
        faces = net.forward()

        for i in range(0, faces.shape[2]):
            confidence = faces[0, 0, i, 2]
            if confidence < 0.5:
                continue
            box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
            start_x, start_y, end_x, end_y = box.astype("int")

            # covnert to square images
            center_x, center_y = (start_x + end_x) // 2, (start_y + end_y) // 2
            square_length = ((end_x - start_x) + (end_y - start_y)) // 2 // 2

            square_length *= 1.1

            start_x = int(center_x - square_length)
            start_y = int(center_y - square_length)
            end_x = int(center_x + square_length)
            end_y = int(center_y + square_length)

            cv2.rectangle(
                frame, (start_x, start_y), (end_x, end_y), (179, 255, 179), 2
            )
        out.write(frame)
 

    end_time = time.time()
    print("Spending time is %s s" % (end_time - start_time))

if __name__ == "__main__":
    main()
