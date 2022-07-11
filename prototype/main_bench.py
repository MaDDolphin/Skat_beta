""" Детектор драк """
# pylint: disable=C0411,E1101,C0103,C0116,W0621,R0914,C0200,R1702,R1703,R1705,R1710,R0912,R0915,W0601
import sys
import cv2
import time
import numpy as np
import argparse
import datetime
import os
import psutil
import GPUtil
import yaml
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

parser = argparse.ArgumentParser(description='Run keypoint detection')
parser.add_argument("--device", default="gpu", help="Device to inference on")
parser.add_argument("--source", default="1.mp4", help="Input Video")

args = parser.parse_args()

protoFile = "pose/coco/pose_deploy_linevec.prototxt"
weightsFile = "pose/coco/pose_iter_440000.caffemodel"
nPoints = 18
keypointsMapping = ['Нос', 'Шея',
                    'П-плечо', 'П-локоть', 'П-кисть',
                    'Л-плечо', 'Л-локоть', 'Л-кисть',
                    'П-бедро', 'П-колено', 'П-стопа',
                    'Л-бедро', 'Л-колено', 'Л-стопа',
                    'П-глаз', 'Л-глаз', 'П-ухо', 'Л-ухо']

POSE_PAIRS = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7],
              [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13],
              [1, 0], [0, 14], [14, 16], [0, 15], [15, 17],
              [2, 17], [5, 16]]

# индекс звеньев, соответствующих POSE_PAIRS
# например, для POSE_PAIR (1,2), они расположены в индексах
# (31,32) вывода, аналогично (1,5) -> (39,40) и так далее.
mapIdx = [[31, 32], [39, 40], [33, 34], [35, 36], [41, 42], [43, 44],
          [19, 20], [21, 22], [23, 24], [25, 26], [27, 28], [29, 30],
          [47, 48], [49, 50], [53, 54], [51, 52], [55, 56],
          [37, 38], [45, 46]]

colors = [[0, 100, 255], [0, 100, 255], [0, 255, 255], [0, 100, 255], [0, 255, 255], [0, 100, 255],
          [0, 255, 0], [255, 200, 100], [255, 0, 255], [0, 255, 0], [255, 200, 100], [255, 0, 255],
          [0, 0, 255], [255, 0, 0], [200, 200, 0], [255, 0, 0], [200, 200, 0], [0, 0, 0]]

input_source = args.source


# Получние точек
def getKeypoints(probMap, threshold=0.1):
    mapSmooth = cv2.GaussianBlur(probMap, (3, 3), 0, 0)

    mapMask = np.uint8(mapSmooth > threshold)
    keypoints = []

    # находим точки
    contours, _ = cv2.findContours(mapMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # для каждой точки находится максимум
    for cnt in contours:
        blobMask = np.zeros(mapMask.shape)
        blobMask = cv2.fillConvexPoly(blobMask, cnt, 1)
        maskedProbMap = mapSmooth * blobMask
        _, maxVal, _, maxLoc = cv2.minMaxLoc(maskedProbMap)
        keypoints.append(maxLoc + (probMap[maxLoc[1], maxLoc[0]],))

    return keypoints


# Поиск правильных связяй между различными суставами всех присутствующих


def getValidPairs(output, frameWidth, frameHeight, detected_keypoints):
    valid_pairs = []
    invalid_pairs = []
    n_interp_samples = 10
    paf_score_th = 0.1
    conf_th = 0.7
    # цикл для каждой POSE_PAIR
    for k in range(len(mapIdx)):
        # A->B составляет конечность
        pafA = output[0, mapIdx[k][0], :, :]
        pafB = output[0, mapIdx[k][1], :, :]
        pafA = cv2.resize(pafA, (frameWidth, frameHeight))
        pafB = cv2.resize(pafB, (frameWidth, frameHeight))

        # Поиск ключевых точкек для первой и второй конечностей
        candA = detected_keypoints[POSE_PAIRS[k][0]]
        candB = detected_keypoints[POSE_PAIRS[k][1]]
        nA = len(candA)
        nB = len(candB)

        # Если ключевые точки для совместной пары обнаружены
        # проверяем каждое соединение в CandA с каждым суставом в CandB
        # Вычислить вектор расстояния между двумя суставами
        # Найдите значения PAF в наборе точек
        # между соединениями
        # Используйте приведенную выше формулу,
        # чтобы вычислить оценку, чтобы отметить соединение как действительное

        if (nA != 0 and nB != 0):
            valid_pair = np.zeros((0, 3))
            for i in range(nA):
                max_j = -1
                maxScore = -1
                found = 0
                for j in range(nB):
                    # Поиск d_ij
                    d_ij = np.subtract(candB[j][:2], candA[i][:2])
                    norm = np.linalg.norm(d_ij)
                    if norm:
                        d_ij = d_ij / norm
                    else:
                        continue
                    # Поиск p(u)
                    interp_coord = list(zip(
                        np.linspace(candA[i][0], candB[j][0], num=n_interp_samples),
                        np.linspace(candA[i][1], candB[j][1], num=n_interp_samples)))
                    # Поиск L(p(u))
                    paf_interp = []
                    for k in range(len(interp_coord)):
                        paf_interp.append([pafA[int(round(interp_coord[k][1])),
                                                int(round(interp_coord[k][0]))],
                                           pafB[int(round(interp_coord[k][1])),
                                                int(round(interp_coord[k][0]))]])
                    # Поиск E
                    paf_scores = np.dot(paf_interp, d_ij)
                    avg_paf_score = sum(paf_scores) / len(paf_scores)

                    # Проверка правильности подключения
                    # Если доля интерполированных векторов,
                    # выровненных с PAF, больше, чем threshold -> Valid Pair
                    if (len(np.where(paf_scores > paf_score_th)[0]) / n_interp_samples) > conf_th:
                        if avg_paf_score > maxScore:
                            max_j = j
                            maxScore = avg_paf_score
                            found = 1
                # Запись соединений в список
                if found:
                    valid_pair = np.append(valid_pair,
                                           [[candA[i][3], candB[max_j][3], maxScore]], axis=0)

            # Добавление обнаруженых соединений в общий список
            valid_pairs.append(valid_pair)
        else:
            invalid_pairs.append(k)
            valid_pairs.append([])
    return valid_pairs, invalid_pairs


# Эта функция создает список ключевых точек, принадлежащих каждому человеку
# Для каждой обнаруженной действительной пары сустав (-ы) назначается человеку

def getPersonwiseKeypoints(valid_pairs, invalid_pairs, keypoints_list):
    # Последняя цифра в каждой строке - балл
    personwiseKeypoints = -1 * np.ones((0, 19))

    for k in range(len(mapIdx)):
        if k not in invalid_pairs:
            partAs = valid_pairs[k][:, 0]
            partBs = valid_pairs[k][:, 1]
            indexA, indexB = np.array(POSE_PAIRS[k])

            for i in range(len(valid_pairs[k])):
                found = 0
                person_idx = -1
                for j in range(len(personwiseKeypoints)):
                    if personwiseKeypoints[j][indexA] == partAs[i]:
                        person_idx = j
                        found = 1
                        break
                if found:
                    personwiseKeypoints[person_idx][indexB] = partBs[i]
                    personwiseKeypoints[person_idx][-1] += \
                        keypoints_list[partBs[i].astype(int), 2] + valid_pairs[k][i][2]

                # если в подмножестве нет partA, создается новое подмножество
                elif not found and k < 17:
                    row = -1 * np.ones(19)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    # добавление keypoint_scores
                    # для двух ключевых точек и paf_score
                    row[-1] = sum(keypoints_list[valid_pairs[k]
                                                 [i, :2].astype(int), 2]) + valid_pairs[k][i][2]
                    personwiseKeypoints = np.vstack([personwiseKeypoints, row])
    return personwiseKeypoints


# Определение угла
def degr(a, b, c):
    ba = a - b
    bc = c - b

    dg = np.degrees(np.arccos(np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))))

    return dg


# Поиск ударов на основе примитивного перцептрона
def check(last, detected_keypoints):
    if last != [] and detected_keypoints != []:
        if last[2] != [] and last[3] != [] and last[4] != []:
            if detected_keypoints[2] != [] \
                    and detected_keypoints[3] != [] and detected_keypoints[4] != []:
                a = np.array([last[2][0][0], last[2][0][1]])
                b = np.array([last[3][0][0], last[3][0][1]])
                c = np.array([last[4][0][0], last[4][0][1]])
                d1 = degr(a, b, c)
                a = np.array([detected_keypoints[2][0][0], detected_keypoints[2][0][1]])
                b = np.array([detected_keypoints[3][0][0], detected_keypoints[3][0][1]])
                c = np.array([detected_keypoints[4][0][0], detected_keypoints[4][0][1]])
                e = np.array([detected_keypoints[2][0][0], 300])
                d2 = degr(a, b, c)
                d2w = degr(b, a, e)

                if 1 <= d1 <= 90 and d2 >= 150 and d2w >= 40:
                    return True
        if last[5] != [] and \
                last[6] != [] and last[7] != []:
            if detected_keypoints[5] != [] and \
                    detected_keypoints[6] != [] and detected_keypoints[7] != []:
                a = np.array([last[5][0][0], last[5][0][1]])
                b = np.array([last[6][0][0], last[6][0][1]])
                c = np.array([last[7][0][0], last[7][0][1]])
                d3 = degr(a, b, c)
                a = np.array([detected_keypoints[5][0][0], detected_keypoints[5][0][1]])
                b = np.array([detected_keypoints[6][0][0], detected_keypoints[6][0][1]])
                c = np.array([detected_keypoints[7][0][0], detected_keypoints[7][0][1]])
                e = np.array([detected_keypoints[5][0][0], 300])
                d4 = degr(a, b, c)
                d4w = degr(b, a, e)

                if 1 <= d3 <= 90 and d4 >= 150 and d4w >= 40:
                    return True
                else:
                    return False


def normilized_timestamp(cap):
    milliseconds = cap.get(cv2.CAP_PROP_POS_MSEC)

    seconds = milliseconds // 1000
    milliseconds = milliseconds % 1000
    minutes = 0
    hours = 0
    if seconds >= 60:
        minutes = seconds // 60
        seconds = seconds % 60

    if minutes >= 60:
        hours = minutes // 60
        minutes = minutes % 60
    right_time = str(hours) + ":" + str(minutes) + ":" + str(round(seconds)) + ":" + str(round(milliseconds))
    return right_time


def bench_stats(cap, t):
    cur_bench = {'time': normilized_timestamp(cap),
                 'cpu': psutil.cpu_percent(),
                 'ram': (psutil.virtual_memory().used / 1048576),
                 'gpuload': GPUtil.getGPUs()[0].load,
                 'fps': (1 / (time.time() - t))}
    return cur_bench


def main():
    input_source_name = os.path.basename(input_source)
    try:
        os.mkdir('./results/' + input_source_name)
    except:
        print('Файл с данным названием уже обрабатывался!')

    # print('Файл с данным названием уже обрабатывался!')
    timestamp = open('./results/' + input_source_name + '/TimeStamps.txt', 'a')
    global i, B, A
    cap = cv2.VideoCapture(input_source)
    hasFrame, frame = cap.read()
    vid_writer = cv2.VideoWriter("./results/" + input_source_name + "/%r.mp4" % (input_source_name + " " +
                                                                                 datetime.datetime.now().strftime(
                                                                                     "%d-%m-%Y %H:%M:%S")),
                                 cv2.VideoWriter_fourcc(*'avc1'), 10,
                                 (frame.shape[1], frame.shape[0]))

    t = time.time()
    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
    if args.device == "cpu":
        net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
        print("Using CPU device")
    elif args.device == "gpu":
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        print("Using GPU device")
    last = []
    bench = []
    while cv2.waitKey(1) < 0:
        t = time.time()
        hasFrame, frame = cap.read()
        frameCopy = np.copy(frame)
        if not hasFrame:
            break

        frameWidth = frame.shape[1]
        frameHeight = frame.shape[0]
        inHeight = 360
        inWidth = int((inHeight / frameHeight) * frameWidth)
        threshold = 0.1

        inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                                        (0, 0, 0), swapRB=False, crop=False)
        net.setInput(inpBlob)
        output = net.forward()

        H = output.shape[2]
        W = output.shape[3]
        # Пустой список для хранения обнаруженных ключевых точек
        points = []
        detected_keypoints = []
        keypoints_list = np.zeros((0, 3))
        keypoint_id = 0
        threshold = 0.1

        for part in range(nPoints):
            # Список достоверности соответствующей части тела
            probMap = output[0, part, :, :]
            probMap = cv2.resize(probMap, (frame.shape[1], frame.shape[0]))
            keypoints = getKeypoints(probMap, threshold)
            keypoints_with_id = []
            for i in range(len(keypoints)):
                keypoints_with_id.append(keypoints[i] + (keypoint_id,))
                keypoints_list = np.vstack([keypoints_list, keypoints[i]])
                keypoint_id += 1

            detected_keypoints.append(keypoints_with_id)
            # Поиск максимума в probMap
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

            # Масштабирование точки, чтобы она соответствовала исходному изображению
            x = (frameWidth * point[0]) / W
            y = (frameHeight * point[1]) / H
            if prob > threshold:
                cv2.circle(frameCopy, (int(x), int(y)), 8,
                           (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.putText(frameCopy, "{}".format(i),
                            (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                            lineType=cv2.LINE_AA)

                # Добавление точки в список, если вероятность больше порога
                points.append((int(x), int(y)))
            else:
                points.append(None)

        frameClone = frame.copy()
        for i in range(nPoints):
            for j in range(len(detected_keypoints[i])):
                cv2.circle(frameClone,
                           detected_keypoints[i][j][0:2], 5, colors[i], -1, cv2.LINE_AA)

        valid_pairs, invalid_pairs = \
            getValidPairs(output, frameWidth, frameHeight, detected_keypoints)
        personwiseKeypoints = \
            getPersonwiseKeypoints(valid_pairs, invalid_pairs, keypoints_list)

        # Отрисовка скелета
        if check(last, detected_keypoints):
            for i in range(17):
                for n in range(len(personwiseKeypoints)):
                    index = personwiseKeypoints[n][np.array(POSE_PAIRS[i])]
                    if -1 in index:
                        continue
                    B = np.int32(keypoints_list[index.astype(int), 0])
                    A = np.int32(keypoints_list[index.astype(int), 1])
                    cv2.line(frameClone, (B[0], A[0]),
                             (B[1], A[1]), (0, 0, 255), 3, cv2.LINE_AA)

            cv2.putText(frameClone, "{:.2f} Кадр/Сек. Драка!".
                        format(1 / (time.time() - t)), (50, 50),
                        cv2.FONT_HERSHEY_COMPLEX,
                        .8,
                        (0, 0, 255), 2, lineType=cv2.LINE_AA)
            cv2.imshow('Обработка', frameClone)
            vid_writer.write(frameClone)
            timestamp.write((normilized_timestamp(cap)) + '\n')

            bench.append(bench_stats(cap, t))

            template = "./results/" + input_source_name + "/{number}.jpg"
            cv2.imwrite(template.format(number=str(normilized_timestamp(cap))), frameClone)

        else:
            for i in range(17):
                for n in range(len(personwiseKeypoints)):
                    index = personwiseKeypoints[n][np.array(POSE_PAIRS[i])]
                    if -1 in index:
                        continue
                    B = np.int32(keypoints_list[index.astype(int), 0])
                    A = np.int32(keypoints_list[index.astype(int), 1])
                    cv2.line(frameClone, (B[0], A[0]), (B[1], A[1]), colors[i], 3, cv2.LINE_AA)

            cv2.putText(frameClone, "{:.2f} Кадр/Сек".format(1 / (time.time() - t)), (50, 50),
                        cv2.FONT_HERSHEY_COMPLEX,
                        .8,
                        (255, 100, 0), 2, lineType=cv2.LINE_AA)
            cv2.imshow('Обработка', frameClone)
            vid_writer.write(frameClone)
            bench.append(bench_stats(cap, t))
        last = detected_keypoints
    with open('./results/' + input_source_name + '_' + args.device + '_bench.txt', 'a') as f:
        yaml.dump(bench, f, default_flow_style=False)
    bench.clear()
    timestamp.close()
    vid_writer.release()
    cv2.destroyAllWindows()
    sys.exit(0)


if __name__ == '__main__':
    main()
