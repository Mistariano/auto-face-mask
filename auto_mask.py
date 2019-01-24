import argparse
import itertools
from util import auto_mask_single_img, imsave, detect, get_mask
import os
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', '-I', dest='input_dir', default='./input')
parser.add_argument('--output_dir', '-O', dest='output_dir', default='./output')

args = parser.parse_args()

eye_ft_list = [
    ['single_eye'],
    ['double_eye'],
    ['l_eye'],
    ['r_eye']
]

mouth_ft_list = [
    ['mouth'],
    ['teeth'],
]

nose_ft_list = [
    ['nose'],
]

prod = lambda x, y: [a + b for a, b in itertools.product(x, y)]

ft_lists = \
    eye_ft_list + \
    mouth_ft_list + \
    nose_ft_list + \
    prod(eye_ft_list, mouth_ft_list) + \
    prod(eye_ft_list, nose_ft_list) + \
    prod(nose_ft_list, mouth_ft_list) + \
    [['all']]

print('ft_lists =', ft_lists)


def main():
    input_dir = args.input_dir

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for f_path in os.listdir(input_dir):
        f = f_path.split('.')
        suffix = f[-1].lower()
        prefix = os.path.split(f[-2])[-1]

        filename = prefix + '.' + suffix

        img = cv2.imread(f_path)
        centers, landmarks = detect(img)

        if landmarks is None:
            with open(os.path.join(output_dir, 'fails.log'), 'w+') as f:
                f.write(filename)
            print(filename, 'failed')
            continue

        print(filename)
        if suffix in ['jpg', 'png']:
            for ft_list in ft_lists:
                dirname = '-'.join(ft_list)
                ft_dir = os.path.join(output_dir, dirname)
                # img = auto_mask_single_img(f_path, ft_list, disturb_ellipse=0)
                mask = get_mask(img, ft_list, centers, landmarks, disturb_ellipse=2, randrange=(10, 50))

                if not os.path.exists(ft_dir):
                    os.makedirs(ft_dir)
                imsave(os.path.join(ft_dir, filename), mask)


if __name__ == '__main__':
    main()
