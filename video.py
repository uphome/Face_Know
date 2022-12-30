import cv2
import time
from tqdm import tqdm
import os
import numpy as np
import sys

#
#
#
####相关设置####
Name = "FENG"
name_set = False  # Ture 为保存脸部信息 False 为不保存脸部信息
data_path = '/home/hu/PycharmProjects/Face_know/Face_data/'  # 脸部数据存储位置


########

def time_kown(times):
    print("\r", end="")
    print("『INFO』进度: {}%: ".format(times), "▓" * (times // 2), end="")
    sys.stdout.flush()
    time.sleep(0.05)


#################################
number_video = 0
count = 0
image_dit = []
labels = []
name_list = {}
url = "http://admin:123456@192.168.69.133:8081"
# video="/home/hu/桌面/home/hu/桌面/VID20221229180719.mp4"


# ！ 导入数据集矩阵
face_cascade = cv2.CascadeClassifier('/home/hu/桌面/haarcascade_frontalface_alt.xml')
face_cascade.load('/home/hu/桌面/haarcascade_frontalface_alt.xml')

print('『INFO』start')
print("『INFO』是否保存人脸信息：" + str(name_set) + '\n')
if name_set:
    print("姓名为" + Name)
    os.mkdir(str(data_path + Name))
else:  # 构造原始人脸数据库   以及人脸标签库
    Name_arry = os.listdir(data_path)
    # print(len(Name_arry))
    for i in range(len(Name_arry)):
        name_list[Name_arry[i]] = i + 1

    for i in Name_arry:
        for j in os.listdir(str(data_path + i)):
            # print(data_path+i+'/'+j)
            image_dit.append(cv2.imread(str(data_path + i + "/" + j), cv2.IMREAD_GRAYSCALE))

            labels.append(name_list[i])

# ! 用现有的数据集训练识别器
print("『INFO』开始训练选择器")
time_kown(0)
recognizer_a = cv2.face.EigenFaceRecognizer_create()  ## EigenFace(PCA，5000以下判断可靠）
recognizer_a.train(image_dit, np.array(labels))  ##  这里只能是整数哟
time_kown(33)
print("  EigenFace 训练完成")
recognizer_b = cv2.face.LBPHFaceRecognizer_create()  # LBPH（局部二值模式直方图，0完全匹配，50以下可接受，80不可靠
recognizer_b.train(image_dit, np.array(labels))
time_kown(66)
print("  LBPH 训练完成")
recognizer_c = cv2.face.FisherFaceRecognizer_create()  # Fisher(线判别分析 ， 5000以下判断为可靠）
recognizer_c.train(image_dit, np.array(labels))
time_kown(100)
print("  Fisher 训练完成")
print("『INFO』选择器训练完成！")

# ! 开始运行
cap = cv2.VideoCapture(url)  # 读取视频流    1280*720
# print("『ERRO』请检查是否打开摄像头 或 是否在同一局域网内")
time.sleep(0.5)  # 停止时间 看清楚相关设置
while cap.isOpened():
    number = 0
    ret, frame = cap.read()
    # print('读取正常\n 现在是第%d帧' % number_video)
    image_ray = cv2.bilateralFilter(frame, 4, 75, 75)

    image_gray = cv2.cvtColor(image_ray, cv2.COLOR_BGR2GRAY)  # 灰度图
    faces = face_cascade.detectMultiScale(image_gray, 1.3, 5)  # 函数检测人脸
    for (x, y, w, h) in faces:
        # 画方框
        img = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # cv2.imshow("img",img)
        # 获取人脸，并转换为200,200的统一格式
        # 原始数据返回的是正方形
        f = cv2.resize(image_gray[y:y + h, x:x + w], (200, 200))  # 脸部图片
        # cv2.imshow("f",f)
        if name_set == False:
            print("\033[1;44m 『INFO』识别成功 \033[1;0m")
        # cv2.imshow("camera", img)
        number = 1
        # 保存脸部图片   可选是否保存脸部数据
        if name_set:
            if ret:
                cv2.imwrite("/home/hu/PycharmProjects/Face_know/Face_data/" + Name + '/%s.png' % str(count), f)
                count = count + 1
        else:
            labels_a, correct_num_a = recognizer_a.predict(f)
        labels_b, correct_num_b = recognizer_b.predict(f)
        labels_c, correct_num_c = recognizer_c.predict(f)
        # print(labels_a,labels_b,labels_c)
        labels = [labels_a, labels_b, labels_c]
        Labels = max(labels, key=labels.count)
        # print(correct_num_a, correct_num_b, correct_num_c)
        correct_num_a = (correct_num_a - 5000)
        correct_num_b = (correct_num_b - 50)
        correct_num_c = (correct_num_c - 5000)
        # print(correct_num_a,correct_num_b,correct_num_c)
        text = list(name_list.keys())[
            list(name_list.values()).index(Labels)]
        # print(x,y)
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), 5)
        print("\033[1;44m 『INFO』识别为 \033[1;0m" + "\033[1;42m%s\033[1;0m" % text)

        cv2.imshow("camera", img)

        # 展示图片
    if number == 0:
        print("\033[1;41m 『INFO』无法识别 \033[1;0m")
        cv2.imshow('camera', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
