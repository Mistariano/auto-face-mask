import argparse
from automask import auto_mask, imsave
import os

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', dest='input_dir', default='./input')
parser.add_argument('--output_dir', dest='output_dir', default='./output')

args = parser.parse_args()


def main():
    input_dir = args.input_dir
    output_dir = args.output_dir
    for f_path in os.listdir(input_dir):
        f = f_path.split('.')
        suffix = f[-1].lower()
        prefix = os.path.split(f[-2])[-1]

        filename = prefix + '.' + suffix

        if suffix in ['jpg', 'png']:
            img = auto_mask(f_path, ['*'])
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            imsave(os.path.join(output_dir, filename), img)


if __name__ == '__main__':
    main()
