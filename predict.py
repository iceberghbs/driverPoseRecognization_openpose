
import cv2 as cv
from cv2 import imshow
import numpy as np

BODY_PARTS = {"Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
              "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
              "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
              "LEye": 15, "REar": 16, "LEar": 17, "Background": 18}

POSE_PAIRS = [["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
              ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
              ["Neck", "Nose"], ["Nose", "REye"], ["REye", "REar"], ["Nose", "LEye"], 
              ["LEye", "LEar"]]

POSES = {0:'driving dangerously', 1:'driving normally', 2:'phone Talking'}
COLORS = {0:(50, 50, 255), 1:(50, 255, 50), 2:(0, 255, 255)}

inWidth = 160  # 输入尺寸
inHeight = 120  # 输入尺寸
inScale = 1.0
frameWidth = 320  # 显示尺寸
frameHeight = 240  # 显示尺寸
thr = 0.1

print()
print('_'*50 + 'BEGIN!' + '_'*50)

# 传入训练结果的模型数据
model_index = 19
model_name = 'model' + str(model_index)
net = cv.dnn.readNetFromTensorflow('./recognize_net/' + model_name + '.pb')  # 传入训练结果的模型数据
net_openpose = cv.dnn.readNetFromTensorflow("./openpose_net/openpose_net.pb")  # 传入openpose的模型数据

# # offline detect
# img = cv.imread('./skeletons/dangerously/img_3661.jpg', cv.IMREAD_GRAYSCALE)
# # cv.imshow('img', img)
# # cv.waitKey(0)

# # blobFromImage 是预处理图像的 + 送入网络得到结果
# inp = cv.dnn.blobFromImage(img, inScale, (inWidth, inHeight),
#                         (0, 0, 0), swapRB=False, crop=False)
# net.setInput(inp)
# out = net.forward()

# # 输出softmax的结果的映射：mnist识别数字的
# # out_list = out[0].tolist()
# # confidence, number_predict = max(out[0]), out_list.index(max(out_list))
# # print()
# # print('the number is: ', number_predict)
# # print('confidence: ', confidence)

# # cv.putText(img, 'the number is: %d' % number_predict, (10, 20), 
# #             cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))  # 可以用这个做实时标志：
# # cv.imshow('number', img)
# # cv.waitKey(0)

# # 输出softmax的结果的映射：识别驾驶员的
# POSES = {0:'driving dangerously', 1:'driving normally', 2:'phone Talking'}
# out_list = out[0].tolist()
# confidence, pose_index = max(out[0]), out_list.index(max(out_list))
# driver_pose = POSES[pose_index]
# print()
# print(out_list)
# print('the driver is: ' + driver_pose)
# print('confidence: ', confidence)

# cv.putText(img, 'The driver is: ' + driver_pose, (10, 20), 
#             cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))  # 可以用这个做实时标志：
# cv.imshow('number', img)
# cv.waitKey(0)



# online detect
cap = cv.VideoCapture(0)  # 如果没有输入图像，那么就调用系统的摄像头 地址：0
while cv.waitKey(1) < 0:  # 这就是按任意键退出
    hasFrame, frame = cap.read(cv.IMREAD_GRAYSCALE)  # 每秒读入一帧图片
    if not hasFrame:
        cv.waitKey()
        break

    imgWidth = frame.shape[1]  # 获取图像的大小信息
    imgHeight = frame.shape[0]
    lineWidth = int(0.03 * imgWidth)  # 根据原图大小等比例画骨架
    circleRadius = int(0.03 * imgWidth)
    inp_openpose = cv.dnn.blobFromImage(frame, inScale, (frameWidth, frameHeight),
                              (0, 0, 0), swapRB=False, crop=False)  
    net_openpose.setInput(inp_openpose)
    out_openpose = net_openpose.forward()
    out_openpose = out_openpose[:, :19, :, :]  # MobileNet output [1, 57, -1, -1], we only need the first 19 elements
    
    points = []
    for i in range(len(BODY_PARTS)):  # 执行像身体部分那么多次的循环（有18部分，但是包括背景有19次，见上面）每个都找出最有可能的关键点
        # Slice heatmap of corresponging body's part.
        heatMap = out_openpose[0, i, :, :]
        _, conf, _, point = cv.minMaxLoc(heatMap)  # 求一个最大值，并返回其索引
        x = (imgWidth * point[0]) / out_openpose.shape[3]  # 建立坐标系，换算在图中的坐标
        y = (imgHeight * point[1]) / out_openpose.shape[2]
        points.append((int(x), int(y)) if conf > thr else None)  # points里面是坐标

    # 画骨架图在实时图像上
    for pair in POSE_PAIRS:
        partFrom = pair[0]  # 起点
        partTo = pair[1]  # 终点
        assert (partFrom in BODY_PARTS)  # if not, raise assertionError
        assert (partTo in BODY_PARTS)

        idFrom = BODY_PARTS[partFrom]  # 起点部位的编号，通过字典的key访问value, key: value
        idTo = BODY_PARTS[partTo]  # 终点部位的编号

        if points[idFrom] and points[idTo]:  # 画图，线和点； 逻辑：如果起点和终点同时都识别了的话，就画点连线
            cv.line(frame, points[idFrom], points[idTo], (50, 255, 50), lineWidth, lineType = cv.LINE_AA)
            cv.circle(frame, points[idFrom], circleRadius, (100, 100, 255), cv.FILLED)
            cv.circle(frame, points[idTo], circleRadius, (50, 50, 255), cv.FILLED)


    crucial_points = points[:4]
    crucial_points.extend(points[5:8])
    if None in crucial_points:
        cv.putText(frame, 'missing crucial points', (400, 20), 
            cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255))
    else:
        # 画骨架图在纯色画布上
        paper = cv.imread('./paper.jpg')
        paper = cv.resize(paper, (imgWidth, imgHeight), interpolation=cv.INTER_AREA)  # 画布设为原图大小
        for pair in POSE_PAIRS:
            partFrom = pair[0]  # 起点
            partTo = pair[1]  # 终点
            assert (partFrom in BODY_PARTS)  # if not, raise assertionError
            assert (partTo in BODY_PARTS)

            idFrom = BODY_PARTS[partFrom]  # 起点部位的编号，通过字典的key访问value, key: value
            idTo = BODY_PARTS[partTo]  # 终点部位的编号

            if points[idFrom] and points[idTo]:  # 画图，线和点； 逻辑：如果起点和终点同时都识别了的话，就画点连线
                cv.line(paper, points[idFrom], points[idTo], (255, 255, 255), lineWidth, lineType = cv.LINE_AA)

        img = cv.cvtColor(paper, cv.COLOR_BGR2GRAY)
        # blobFromImage 是预处理图像的
        inp = cv.dnn.blobFromImage(img, inScale, (inWidth, inHeight),
                                (0, 0, 0), swapRB=False, crop=False)

        net.setInput(inp)
        out = net.forward()
        
        # 结果映射
        out_list = out[0].tolist()
        confidence, pose_index = max(out[0]), out_list.index(max(out_list))
        driver_pose = POSES[pose_index]
        font_color = COLORS[pose_index]
        
        # 实时显示结果
        cv.putText(frame, 'The driver is: ' + driver_pose, (10, 20), 
                    cv.FONT_HERSHEY_SIMPLEX, 0.6, font_color)
        cv.putText(frame, 'Confidence : ' + str(confidence), (10, 40), 
                    cv.FONT_HERSHEY_SIMPLEX, 0.6, (50, 255, 50))

    cv.imshow('frame', frame)


print()
print('_'*50 + 'ALL DONE!' + '_'*50)
print()
