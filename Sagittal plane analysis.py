import cv2 as cv2
import numpy as np
from subprocess import call
from math import ceil, sqrt, pi
import matplotlib.pyplot as plt


def get_angle(x_1, x_2, x_3, y_1, y_2, y_3):
    cos = ((x_1-x_2)*(x_1-x_3)+(y_1-y_2)*(y_1-y_3)) / (sqrt((x_1-x_2)**2+(y_1-y_2)**2) * sqrt((x_1-x_3)**2+(y_1-y_3)**2))
    angle = np.arccos(cos)*180/pi
    return angle


def template_demo(tpl, target, method=cv2.TM_CCORR_NORMED):
    th, tw = tpl.shape[:2]  # 取高宽，不取通道 模板高宽
    result = cv2.matchTemplate(target, tpl, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)  # 寻找最小值，最大值。最小值位置，最大值位置
    print('max_val=', max_val)
    tl = max_loc
    br = (tl[0] + tw, tl[1] + th)
    print(max_val, tl, br)
    if max_val < 0.45:
        lost = 1
    else:
        lost = 0
    return tl, br, lost


if __name__ == '__main__':

    number = int(input('please input the number of objects: '))
    area = np.zeros((number, 4))
    ROI_area = np.zeros((number, 4))
    print('area.shape=', area.shape)
    print("--------- Python OpenCV Tutorial ---------")
    cap = cv2.VideoCapture('901285474.mp4')
    ret, frame = cap.read()
    original_frame = frame
    print('frame.shape', frame.shape)
    lam = 1
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 保存视频的编码
    out = cv2.VideoWriter('_output.avi', fourcc, 20.0, (frame.shape[1], frame.shape[0]))
    for i in range(number):
        gROI = cv2.selectROI("ROI frame", frame, False)
        ROI_area[i] = [gROI[1], gROI[1] + gROI[3], gROI[0], gROI[0] + gROI[2]]
        # ROI = frame[gROI[1]: gROI[1]+gROI[3], gROI[0]:gROI[0]+gROI[2], :]
        area[i] = [0, frame.shape[1], 0, frame.shape[0]]
    print('area=', area)
    print('ROI_area=', ROI_area)
    Hip_Angle, Knee_Angle, NFA, UFA = [], [], [], []
    while True:
        ret, frame = cap.read()
        if ret:
            plate_x, plate_y, hip_x, hip_y, knee_x, knee_y, ankle_x, ankle_y = [], [], [], [], [], [], [], []

            for k in range(number):
                ROI = original_frame[int(ROI_area[k, 0]): int(ROI_area[k, 1]), int(ROI_area[k, 2]):int(ROI_area[k, 3]), :]
                frame1 = frame[int(area[k, 2]):int(area[k, 3]), int(area[k, 0]):int(area[k, 1]), :]
                tl, br, lost = template_demo(ROI, frame1, method=cv2.TM_CCOEFF_NORMED)
                if lost == 1:
                    lam = 2
                else:
                    lam = 1
                # ROI = frame1[tl[1]:br[1], tl[0]:br[0], :]
                cv2.imshow('ROI=', ROI)
                print('ROI.shape=', ROI.shape)

                high_x = int(area[k, 0]) + tl[0]
                high_y = int(area[k, 2]) + tl[1]
                low_x = br[0] + int(area[k, 0])
                low_y = br[1] + int(area[k, 2])
                result = cv2.rectangle(frame, (high_x, high_y), (low_x, low_y), (k * 20, k * 10, 255 // (k + 1)), 2)#(img, 顶点, 顶点, color, thickness)
                result = cv2.circle(result, (ceil((high_x+low_x)/2), ceil((high_y+low_y)/2)), 8, (0, 0, 255), 0)#绘制圆点
                print('k=', k, high_x, high_y)
                print('k=', k, low_x, low_y)
                if k == 0:
                    plate_x.append(ceil((high_x+low_x)/2))
                    plate_y.append(ceil((high_y+low_y)/2))
                if k == 1:
                    hip_x.append(ceil((high_x+low_x)/2))
                    hip_y.append(ceil((high_y+low_y)/2))
                if k == 2:
                    knee_x.append(ceil((high_x+low_x)/2))
                    knee_y.append(ceil((high_y+low_y)/2))
                if k == 3:
                    ankle_x.append(ceil((high_x+low_x)/2))
                    ankle_y.append(ceil((high_y+low_y)/2))

                    result = cv2.line(result, (plate_x[-1], plate_y[-1]), (plate_x[-1], plate_y[-1]-20), color=(255, 0, 0), thickness=2)#不良力臂
                    result = cv2.line(result, (plate_x[-1], plate_y[-1]-20), (low_x, plate_y[-1]-20), color=(255, 0, 0), thickness=2)
                    result = cv2.line(result, (low_x, low_y), (low_x, plate_y[-1]-20), color=(255, 0, 0), thickness=2, lineType=8)
                    result = cv2.line(result, (plate_x[-1], plate_y[-1]-20), (hip_x[-1], plate_y[-1]-20), color=(255, 0, 0), thickness=2)#必要力臂
                    result = cv2.line(result, (hip_x[-1], plate_y[-1]-20), (hip_x[-1], hip_y[-1]), color=(255, 0, 0), thickness=2)

                area[k, 0] = (tl[0] + br[0]) // 2 - lam * (br[0] - tl[0]) + area[k, 0]
                area[k, 1] = (tl[0] + br[0]) // 2 + lam * (br[0] - tl[0]) + area[k, 0]
                area[k, 2] = (tl[1] + br[1]) // 2 - lam * (br[1] - tl[1]) + area[k, 2]
                area[k, 3] = (tl[1] + br[1]) // 2 + lam * (br[1] - tl[1]) + area[k, 2]
                area[k, 0] = 0 if area[k, 0] < 0 else area[k, 0]
                area[k, 1] = frame.shape[1] if area[k, 1] > frame.shape[1] else area[k, 1]
                area[k, 2] = 0 if area[k, 2] < 0 else area[k, 2]
                area[k, 3] = frame.shape[0] if area[k, 3] > frame.shape[0] else area[k, 3]
            Hip_Angle.append(get_angle(hip_x[-1], plate_x[-1], knee_x[-1], hip_y[-1], plate_y[-1], knee_y[-1]))
            Knee_Angle.append(get_angle(knee_x[-1], hip_x[-1], ankle_x[-1], knee_y[-1], hip_y[-1], ankle_y[-1]))
            NFA.append(abs(hip_x[-1]-low_x))
            UFA.append(abs(plate_x[-1]-low_x))
            out.write(result)
            cv2.waitKey(18)
            cv2.imshow('result', result)
        else:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    print('len Hip_Angle:', len(Hip_Angle))
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(0, len(Hip_Angle), 1), Hip_Angle)
    plt.show()
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(0, len(Knee_Angle), 1), Knee_Angle)
    plt.show()
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(0, len(NFA), 1), NFA)
    plt.show()
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(0, len(UFA), 1), UFA)
    plt.show()

    cap.release()
    cv2.destroyAllWindows()
    command = "ffmpeg -i _output.avi _output.mp4"
    call(command.split())
