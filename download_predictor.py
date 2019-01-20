import wget
import bz2

bz2_filename = 'shape_predictor_68_face_landmarks.dat.bz2'
dat_filename = 'shape_predictor_68_face_landmarks.dat'


def download():
    wget.download(
        'https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2',
        bz2_filename)
    print('Downloaded.')


def unzip():
    with bz2.open(bz2_filename, 'r') as bz2_f:
        dec = bz2_f.read()
    print('Decompressed.')
    with open(dat_filename, 'wb') as f:
        f.write(dec)


if __name__ == '__main__':
    download()
    unzip()
