import wget
import tarfile
from bz2 import BZ2File

bz2_filename = 'shape_predictor_68_face_landmarks.dat.bz2'
dat_filename = 'shape_predictor_68_face_landmarks.dat'


def download():
    wget.download(
        'https://zh.osdn.net/projects/sfnet_dclib/downloads/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2/',
        bz2_filename)
    tar = tarfile.open(bz2_filename)
    tar.extractall()
    tar.close()


def unzip():
    dec = BZ2File(bz2_filename,'r').read(-1)
    with open(dat_filename, 'wb') as f:
        f.write(dec)


if __name__ == '__main__':
    download()
