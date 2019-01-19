import argparse
from automask import detect,add_mask


def main():
    img = None
    centers = detect(img)
    add_mask(img, centers)


if __name__ == '__main__':
    main()