import cv2

number_video = 0
count = 0
url = "http://admin:123456@192.168.69.133:8081"
#video="/home/hu/桌面//home/hu/桌面/VID20221229180719.mp4"

face_cascade =cv2.CascadeClassifier('/home/hu/桌面/haarcascade_frontalface_alt.xml')   #导入opencv 训练好的数据集
face_cascade.load('/home/hu/桌面/haarcascade_frontalface_alt.xml')

print('start')
cap = cv2.VideoCapture(url)  # 读取视频流    1280*720
while cap.isOpened():
    number=0
    ret, frame = cap.read()
    #print('读取正常\n 现在是第%d帧' % number_video)
    image_ray = cv2.bilateralFilter(frame, 4, 75, 75)

    image_gray = cv2.cvtColor(image_ray, cv2.COLOR_BGR2GRAY)  # 灰度图
    faces = face_cascade.detectMultiScale(image_gray, 1.3, 5)  # 函数检测人脸
    for (x, y, w, h) in faces:
        # 画方框
        img = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        #cv2.imshow("img",img)
        # 获取人脸，并转换为200,200的统一格式
        # 原始数据返回的是正方形
        f = cv2.resize(image_ray[y:y + h, x:x + w], (200, 200))  #脸部图片
        #cv2.imshow("f",f)
        print("\033[1;44m 识别成功 \033[1;0m")
        cv2.imshow("camera",img)
        number=1
        # 保存图片
    if ret:
        #cv2.imwrite('/home/hu/PycharmProjects/Face_know/hujaingtao/%s.pgm' % count, frame)
        count += 1
    else:
        break

        # 展示图片
    if number==0:
      print("\033[1;41m 无法识别 \033[1;0m")
      cv2.imshow('camera', frame)



    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    number_video = number_video + 1
cap.release()
cv2.destroyAllWindows()
