import glob
from shutil import copy2
import os


def read_dir_and_copy(dir, num, des):
    images_list = glob.glob(dir + '/*JPEG')
    if len(images_list) == 0:
        print(dir + " has not jpeg")
    for i in range(num):
        image = images_list[i]
        copy2(image, des)


def main():
    dirs = ['n01440764', 'n01443537', 'n01484850', 'n01491361', 'n01494475', 'n01496331', 'n01498041', 'n01514668',
              'n01514859', 'n01518878']
    data_dir = 'data/'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    for dir in dirs:
        new_dir = data_dir + dir + '/'
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        read_dir_and_copy(dir, 30, new_dir)

main()


