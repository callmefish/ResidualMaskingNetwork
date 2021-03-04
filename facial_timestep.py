import os
import glob
import json
import cv2
import numpy as np
import torch
from torchvision.transforms import transforms
from models import resmasking_dropout1
import time


def ensure_color(image):
    if len(image.shape) == 2:
        return np.dstack([image] * 3)
    elif image.shape[2] == 1:
        return np.dstack([image] * 3)
    return image


net = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel"
)


transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])


FER_2013_EMO_DICT = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral",
}

with open('JSON_format.json') as f:
    JSON_templete = json.load(f)

result_path = '/content/drive/MyDrive/youtube_data/saved/result_img/'
video_path = '/content/drive/MyDrive/youtube_data/video/'
output_json = '/content/drive/MyDrive/youtube_data/CSCE636Spring2021-ZhiyuYan-1.json'


def main():
    # load configs and set random seed
    configs = json.load(open("./configs/fer2013_config.json"))
    image_size = (configs["image_size"], configs["image_size"])

    model = resmasking_dropout1(in_channels=3, num_classes=7)
    model.cuda()

    # state = torch.load('./saved/checkpoints/densenet121_rot30_2019Nov11_14.23')
    # state = torch.load('./saved/checkpoints/resmasking_dropout1_rot30_2019Nov17_14.33')
    state = torch.load(
        "./saved/checkpoints/resmasking_dropout1_00"
    )
    model.load_state_dict(state["net"])
    model.eval()

    video_list = os.listdir(video_path)
    print(video_list)

    timestep = {}
    startTime = time.time()
    result = []

    for video_name in video_list:
        print("The video is " + video_name)
        video_item_path = video_path + video_name
        vid = cv2.VideoCapture(video_item_path)

        if not vid.isOpened():
            continue
        else:
            rate = round(vid.get(5))
            print(type(rate))
            FrameNumber = vid.get(7)
            duration = FrameNumber / rate
            step = rate // 8
            print("The duration is %f s" % duration)
            print("The number of frame is %d " % FrameNumber)
            print("The rate of video is %d " % rate)

        subStartTime = time.time()
        
        with torch.no_grad():
            cnt = 0
            happy_time = 0
            happy_Confidence = []
            while True:
                ret, frame = vid.read()
                if frame is None or ret is not True:
                    break
                if cnt % step != 0:
                    cnt += 1
                    continue
                
                frame = np.fliplr(frame).astype(np.uint8)
                # frame += 50
                h, w = frame.shape[:2]
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # gray = frame

                blob = cv2.dnn.blobFromImage(
                    cv2.resize(frame, (300, 300)),
                    1.0,
                    (300, 300),
                    (104.0, 177.0, 123.0),
                )
                net.setInput(blob)
                faces = net.forward()
                
                happy_proba = 0
                for i in range(0, faces.shape[2]):
                    confidence = faces[0, 0, i, 2]
                    if confidence < 0.7:
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
          
                    face = gray[start_y:end_y, start_x:end_x]
                    if face.shape[0] == 0 or face.shape[1] == 0:
                        continue

                    face = ensure_color(face)

                    face = cv2.resize(face, image_size)
                    face = transform(face).cuda()
                    face = torch.unsqueeze(face, dim=0)

                    output = torch.squeeze(model(face), 0)
                    proba = torch.softmax(output, 0)
                    proba = proba.cpu().numpy()

                    # happy[cnt//6] = max(happy[cnt//6], proba[3])

                    emo_proba, emo_idx = proba[3], 3
                    happy_proba = max(happy_proba, emo_proba)

                    emo_label = FER_2013_EMO_DICT[emo_idx]

                    label_size, base_line = cv2.getTextSize(
                        "{}: 000".format(emo_label), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
                    )

                    cv2.rectangle(
                        frame,
                        (end_x, start_y + 1 - label_size[1]),
                        (end_x + label_size[0], start_y + 1 + base_line),
                        (223, 128, 255),
                        cv2.FILLED,
                    )
                    cv2.putText(
                        frame,
                        "{} {}".format(emo_label, int(emo_proba * 100)),
                        (end_x, start_y + 1),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 0),
                        2,
                    )

                video_name_sub = video_name.split('.')[0]
                if happy_proba > 0.8:
                    happy_time += 1
                    happy_Confidence.append(happy_proba)
                    if happy_time > 4:
                        happy_time = 0
                        tmp_json = {}
                        for key, val in JSON_templete.items():
                            tmp_json[key] = val
                        tmp_json["videoId"] = video_name_sub
                        tmp_json["endTime"] = cnt / rate
                        if tmp_json["endTime"] < 2.0:
                            tmp_json["endTime"] = 2.0
                        tmp_json["startTime"] = tmp_json["endTime"] - 2
                        happy_Confidence.sort(reverse=True)
                        tmp_json["observation"]["labelConfidence"] = sum(happy_Confidence[:5])/5
                        result.append(tmp_json)
                        happy_Confidence = []
                    elif len(happy_Confidence) == 8:
                        happy_Confidence = []
                    cv2.imwrite(result_path + video_name + '/' + str(cnt).zfill(5) + '.jpg', frame)
                cnt += 1

        subEndTime = time.time()
        print("Spending time of %s is %s s" % (video_name, subEndTime - subStartTime))
    endTime = time.time()
    print("Total spending time is %s s" % (endTime - startTime))

    with open(output_json, 'w') as  f:
        json.dump(result, f)

if __name__ == "__main__":
    main()
