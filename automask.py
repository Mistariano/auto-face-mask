import dlib
import numpy as np
import cv2
import random

# https://www.cnblogs.com/AdaminXie/p/8137580.html

# 0: right eye
# 1: left eye
# 2: mouth
# 3: nose
ft_idx_range = [(36, 41), (42, 47), (48, 67), (27, 35)]


def detect(img_rd, debug=False):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    img_gray = cv2.cvtColor(img_rd, cv2.COLOR_RGB2GRAY)

    faces = detector(img_gray, 0)

    font = cv2.FONT_HERSHEY_SIMPLEX

    if len(faces) == 1:
        ft_centers = [(0, 0)] * len(ft_idx_range)
        face = faces[0]
        landmarks = np.matrix([[p.x, p.y] for p in predictor(img_rd, face).parts()])
        for idx, point in enumerate(landmarks):

            pos = (point[0, 0], point[0, 1])

            if debug:
                # display feature points
                cv2.circle(img_rd, pos, 2, color=(139, 0, 0))
                cv2.putText(img_rd, str(idx), pos, font, 0.2, (187, 255, 255), 1, cv2.LINE_AA)

            for ft in range(len(ft_idx_range)):
                if ft_idx_range[ft][0] <= idx <= ft_idx_range[ft][1]:
                    ft_centers[ft] = tuple(
                        ft_centers[ft][i] + pos[i] // (ft_idx_range[ft][1] - ft_idx_range[ft][0] + 1) for i in
                        (0, 1))
        for ft in range(len(ft_idx_range)):
            if debug:
                cv2.circle(img_rd, ft_centers[ft], 6, color=(255, 0, 0), thickness=-1)
        return ft_centers

    else:
        msg = 'no faces' if not len(faces) else 'mul faces: %d' % len(faces)
        cv2.putText(img_rd, msg, (20, 50), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        return None


def add_mask(img_rd, ft_centers: list):
    for ft in range(len(ft_idx_range)):
        cv2.ellipse(img_rd, ft_centers[ft], (random.randrange(25, 100), random.randrange(25, 100)),
                    random.randrange(0, 90), 0, 360, (255, 255, 255), thickness=-1)
        cv2.ellipse(img_rd, ft_centers[ft], (random.randrange(25, 100), random.randrange(25, 100)),
                    random.randrange(0, 90), 0, 360, (255, 255, 255), thickness=-1)


if __name__ == '__main__':
    img = cv2.imread("test.jpg")
    centers = detect(img, True)
    add_mask(img, centers)
    cv2.imshow('', img)
    cv2.waitKey(0)
