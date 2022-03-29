# To use Inference Engine backend, specify location of plugins:
# export LD_LIBRARY_PATH=/opt/intel/deeplearning_deploymenttoolkit/deployment_tools/external/mklml_lnx/lib:$LD_LIBRARY_PATH
import cv2 as cv
import numpy as np
import argparse
import os

print()
print('_'*50 + 'BEGIN!' + '_'*50)


parser = argparse.ArgumentParser()
parser.add_argument('--input', help='Path to image or video. Skip to capture frames from camera')
parser.add_argument('--thr', default=0.1, type=float, help='Threshold value for pose parts heat map')  # 置信度，越高识别越不容易
parser.add_argument('--width', default=320, type=int, help='Resize input to specific width.')
parser.add_argument('--height', default=240, type=int, help='Resize input to specific height.')
parser.add_argument('--scale', default=1.0, type=float, help='Scale for blob.')

args = parser.parse_args()

BODY_PARTS = {"Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
              "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
              "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
              "LEye": 15, "REar": 16, "LEar": 17, "Background": 18}

# POSE_PAIRS = [["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
#               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
#               ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
#               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
#               ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]]

POSE_PAIRS = [["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
              ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
              ["Neck", "Nose"], ["Nose", "REye"], ["REye", "REar"], ["Nose", "LEye"], 
              ["LEye", "LEar"]]

inWidth = args.width
inHeight = args.height
inScale = args.scale

net = cv.dnn.readNetFromTensorflow("./openpose_net/openpose_net.pb")  # 传入训练结果的模型数据

if __name__ == '__main__':
    classList = os.listdir('./datasets/')
    for className in classList:
        filedir = './datasets/' + className
        if os.path.isfile(filedir):
            continue
        for filename in os.listdir(filedir):
            img = cv.imread(filedir + '/' + filename)  # opcv format: B G R

            # 判断不是一个文件夹
            if os.path.isdir(filedir + filename):
                continue

            # test
            # cv.imshow('image', img)
            # cv.waitKey(0)

            imgWidth = img.shape[1]  # 获取图像的大小信息
            imgHeight = img.shape[0]
            lineWidth = int(0.03 * imgWidth)  # 根据原图大小等比例画骨架
            circleRadius = int(0.03 * imgWidth)
            # imgMean = cv.mean(img)

            print()
            print('processing:  ' + filename)

            # blobFromImage 是预处理图像的
            inp = cv.dnn.blobFromImage(img, inScale, (inWidth, inHeight),
                                    (0, 0, 0), swapRB=False, crop=False)  
            net.setInput(inp)
            # net.setInput(cv.dnn.blobFromImage(img, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))

            out = net.forward()
            out = out[:, :19, :, :]  # MobileNet output [1, 57, -1, -1], we only need the first 19 elements
            # out is a (1,19,46,46)_tensor ， 有十八个加一个背景十九个要素，每个是一个；46*46是图片大小

            out_reshape = np.reshape(out, (-1, 1))  # 一列的向量
            length_out = len(out_reshape)

            # assert (len(BODY_PARTS) == out.shape[1])
            points = []  # 热力图（confident_map)判断关键点并连线, points是一个list，装的是tuple，是坐标
            for i in range(len(BODY_PARTS)):  # 执行像身体部分那么多次的循环（有18部分，但是包括背景有19次，见上面）每个都找出最有可能的关键点
                # Slice heatmap of corresponging body's part.
                heatMap = out[0, i, :, :]

                # Originally, we try to find all the local maximums. To simplify a sample  找热图的最热点，最有可能的关节点
                # we just find a global one. However only a single pose at the same time    这样一次只能探测一个人
                # could be detected this way.
                _, conf, _, point = cv.minMaxLoc(heatMap)  # 求一个最大值，并返回其索引
                x = (imgWidth * point[0]) / out.shape[3]  # 建立坐标系，换算在图中的坐标
                y = (imgHeight * point[1]) / out.shape[2]
                # Add a point if it's confidence is higher than threshold.  #如果热力比阈值高，标注是关键点
                points.append((int(x), int(y)) if conf > args.thr else None)  # points里面是坐标


            print('the points is:')
            print(points)

            crucial_points = points[:4]
            crucial_points.extend(points[5:8])

            if None in crucial_points:
                print('missing crucial points')
                continue

            # 画骨架图在原图上
            # for pair in POSE_PAIRS:
            #     partFrom = pair[0]  # 起点
            #     partTo = pair[1]  # 终点
            #     assert (partFrom in BODY_PARTS)  # if not, raise assertionError
            #     assert (partTo in BODY_PARTS)

            #     idFrom = BODY_PARTS[partFrom]  # 起点部位的编号，通过字典的key访问value, key: value
            #     idTo = BODY_PARTS[partTo]  # 终点部位的编号

            #     if points[idFrom] and points[idTo]:  # 画图，线和点； 逻辑：如果起点和终点同时都识别了的话，就画点连线
            #         cv.line(img, points[idFrom], points[idTo], (0, 255, 0), lineWidth, lineType = cv.LINE_AA)
            #         cv.circle(img, points[idFrom], circleRadius, (0, 0, 255), cv.FILLED)
            #         cv.circle(img, points[idTo], circleRadius, (0, 0, 255), cv.FILLED)

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
                    # cv.circle(paper, points[idFrom], circleRadius, (255, 255, 255), cv.FILLED)
                    # cv.circle(paper, points[idTo], circleRadius, (255, 255, 255), cv.FILLED)

            # 结果保存
            paper = cv.resize(paper, (inWidth, inHeight), interpolation=cv.INTER_AREA)  # 输出图像尺寸规范化
            results_list = './skeletons/'+ className
            if not os.path.exists(results_list):
                os.mkdir(results_list)
            ok = cv.imwrite(results_list + '/' + filename, paper)
            print('_'*10 + 'success' if ok else '_'*10 + 'fail')
            print()

            # 结果呈现 
            # img = cv.resize(img, (100, 100), interpolation=cv.INTER_CUBIC)
            # cv.imshow('OpenPose using OpenCV', img)
            # cv.waitKey(0)

    print('_'*50 + 'ALL DONE!' + '_'*50)
    print()


## This part is used to detect ONLINE

# cap = cv.VideoCapture(args.input if args.input else 0)  # 如果没有输入图像，那么就调用系统的摄像头 地址：0


# while cv.waitKey(1) < 0:  # 这就是按任意键退出
#     hasFrame, frame = cap.read()  # 每秒读入一帧图片
#     if not hasFrame:
#         cv.waitKey()
#         break

    # frameWidth = frame.shape[1]  # 获取图像的大小信息
    # frameHeight = frame.shape[0]

    # # blobFromImage 是预处理图像的
    # net.setInput(cv.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    # out = net.forward()
    # out = out[:, :19, :, :]  # MobileNet output [1, 57, -1, -1], we only need the first 19 elements
    # # out is a (1,19,46,46)_tensor ， 有十八个加一个背景十九个要素，每个是一个；46*46是图片大小
    # out_reshape = np.reshape(out, (-1, 1))
    # length_out = len(out_reshape)

    # assert (len(BODY_PARTS) == out.shape[1])

    # points = []  # 热力图（confident_map)判断关键点并连线, points是一个list，装的是tuple，是坐标
    # for i in range(len(BODY_PARTS)):  # 执行像身体部分那么多次的循环（有18部分，但是包括背景有19次，见上面）每个都找出最有可能的关键点
    #     # Slice heatmap of corresponging body's part.
    #     heatMap = out[0, i, :, :]

    #     # Originally, we try to find all the local maximums. To simplify a sample  找热图的最热点，最有可能的关节点
    #     # we just find a global one. However only a single pose at the same time    这样一次只能探测一个人
    #     # could be detected this way.
    #     _, conf, _, point = cv.minMaxLoc(heatMap)  # 求一个最大值，并返回其索引
    #     x = (frameWidth * point[0]) / out.shape[3]  # 建立坐标系，换算在图中的坐标
    #     y = (frameHeight * point[1]) / out.shape[2]
    #     # Add a point if it's confidence is higher than threshold.  #如果热力比阈值高，标注是关键点
    #     points.append((int(x), int(y)) if conf > args.thr else None)  # 置信度高于阈值

    # for pair in POSE_PAIRS:
    #     partFrom = pair[0]  # 起点
    #     partTo = pair[1]  # 终点
    #     assert (partFrom in BODY_PARTS)  # if not, raise assertionError
    #     assert (partTo in BODY_PARTS)

    #     idFrom = BODY_PARTS[partFrom]  # 起点部位的编号，通过字典的key访问value, key: value
    #     idTo = BODY_PARTS[partTo]  # 终点部位的编号

    #     if points[idFrom] and points[idTo]:  # 画图，线和点； 逻辑：如果起点和终点同时都识别了的话，就画点连线
    #         cv.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
    #         cv.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)  # 画椭圆函数，也可以画圆
    #         cv.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)

    # t, _ = net.getPerfProfile()
    # freq = cv.getTickFrequency() / 1000
    # cv.putText(frame, '%.2fms' % (t / freq), (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))  # 可以用这个做实时标志：

    # cv.imshow('OpenPose using OpenCV', frame)
