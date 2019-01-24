import dlib
import numpy as np
import cv2
import random

# https://www.cnblogs.com/AdaminXie/p/8137580.html

# 0: right eye
# 1: left eye
# 2: mouth
# 3: nose
ft_idx_range = [(36, 41), (42, 47), (48, 59), (27, 35), (60, 67)]
ft_map = {'r_eye': 0, 'l_eye': 1, 'mouth': 2, 'nose': 3, 'teeth': 4}


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
        return ft_centers, landmarks

    else:
        msg = 'no faces' if not len(faces) else 'mul faces: %d' % len(faces)
        cv2.putText(img_rd, msg, (20, 50), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        return None, None


def get_mask(img_rd, ft_list: list, ft_centers: list, landmarks: list, disturb_ellipse: int = 2,
             randrange: tuple = (25, 100)):
    """
    Add masks.

    :param img_rd: Image object
    :param ft_list: the features need to be masked. e.g. ['r_eye','l_eye','mouth','nose'] or ['all'] or ['*'] or ['single_eye']
    :param ft_centers: center points
    :param landmarks: lm points
    :param disturb_ellipse:
    :param randrange:
    :return: img_rd
    """
    ft_list = [ft for ft in ft_list]
    try:
        assert ft_list in (['*'], ['all']) or set(ft_list).issubset(
            set(ft_map.keys()).union(['single_eye', 'double_eye']))
    except AssertionError as e:
        print('keys:', ft_map.keys())
        raise e

    if 'single_eye' in ft_list:
        ft_list.remove('single_eye')
        if 'l_eye' not in ft_list and 'r_eye' not in ft_list:
            ft_list.append(random.choice(['l_eye', 'r_eye']))
    if 'double_eye' in ft_list:
        ft_list.remove('double_eye')
        ft_list += ['l_eye', 'r_eye']

    if ft_list in (['*'], ['all']):
        ft_list = [name for name in ft_map.keys()]

    ft_set = set([ft_map[ft] for ft in ft_list])

    img_white = np.zeros(img_rd.shape, np.uint8)
    for ft in range(len(ft_idx_range)):
        if ft not in ft_set:
            continue
        ft_st, ft_ed = ft_idx_range[ft]
        lm = np.array(landmarks[ft_st:ft_ed + 1])
        lm = lm.reshape([-1, 1, 2])
        cv2.fillPoly(img_white, [lm], (255, 255, 255))

        rand_min, rand_max = randrange
        for _ in range(disturb_ellipse):
            _a = random.randrange(rand_min, rand_max)
            _b = random.randrange(rand_min, rand_max)
            _angle = random.randrange(0, 90)
            cv2.ellipse(img_white, ft_centers[ft], (_a, _b), _angle, 0, 360, (255, 255, 255), thickness=-1)
    return img_white


def auto_mask_single_img(input_path: str, ft_list: list, debug=False, disturb_ellipse: int = 2, randrange=(25, 100)):
    img = cv2.imread(input_path)
    centers, landmarks = detect(img, debug)
    mask = get_mask(img, ft_list, centers, landmarks, disturb_ellipse=disturb_ellipse, randrange=randrange)
    return mask


def imsave(output_dir, img):
    cv2.imwrite(output_dir, img)


if __name__ == '__main__':
    img = auto_mask_single_img('img.jpg', ['mouth'], True)
    cv2.imshow('', img)
    cv2.waitKey(0)
