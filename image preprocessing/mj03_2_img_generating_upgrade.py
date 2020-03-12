""" 이미지 부풀리기"""

import glob
from keras_preprocessing.image import ImageDataGenerator , load_img , img_to_array
from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Conv2D , MaxPool2D , Flatten , Dense
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from PIL import Image
import numpy as np
import os
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

def img_generating_2(main_dir, categories, n=3):
    """ img_generating 업그레이드 버전 :
    - 파일크기가 큰 이미지를 부풀릴 때 속도가 오래 걸리는 것을 방지하기 위하여 픽셀을 비율에 맞게 축소 후 부풀림.
    - 파일크기가 250KB 이상인 이미지는 픽셀사이즈를 224 비율에 맞게 줄여 부풀리고,
    - 250KB 이하는 64 비율에 맞게 줄여 n만큼 부풀림 """
    datagen = ImageDataGenerator(rescale=1. / 255,
                                 rotation_range=30,  # 각도 범위내 회전
                                 width_shift_range=0.1,  # 수평방향
                                 height_shift_range=0.1,  # 수직방향
                                 brightness_range=[0.2, 1.5],  # 밝기
                                 shear_range=0.7,  # 시계반대방향
                                 zoom_range=[0.8, 1.0], horizontal_flip=True,  # 수평방향 뒤집기
                                 vertical_flip=True, fill_mode='nearest')

    for cat in categories:
        cat_dir = main_dir + cat   # 각 카테고리별 디렉토리
        files = glob.glob(cat_dir + "/*.jpg")   # cat_dir의 jpg 파일들 다 가져옴

        for file in files:
            im = load_img(file)
            im_size = os.stat(file)
            file_size = im_size.st_size / 1024  # KB로 변환
            print(F'file_size: {file_size} KB')
            pixel1 = 224, 224
            pixel2 = 64, 64

            if file_size >= 250:  # 이미지 파일 사이즈
                print(im.size, '-')   # im.size는 픽셀을 말함
                im.thumbnail(pixel1, Image.ANTIALIAS)
                im = im.convert('RGB')
                print(im.size)
                x = img_to_array(im)
                x = x.reshape((1,) + x.shape)
                # x 전체를 하나의 []로 묶음 -> 뒤에서 Conv2D할 때 input이 4차원형태로 들어가야 되기 때문에 미리 변환해놓음
            else:
                print(im.size, '-')   # im.size는 픽셀을 말함
                im.thumbnail(pixel2, Image.ANTIALIAS)
                im = im.convert('RGB')
                print(im.size)
                x = img_to_array(im)
                x = x.reshape((1,) + x.shape)

            i = 0
            for _ in datagen.flow(x, batch_size=1, save_to_dir=cat_dir,
                                  save_prefix=os.path.basename(file).split('.')[0]+'_copy', save_format='jpg'):  # file에서 맨마지막 부분인 tsu0.jpg를 출력하고, '.'으로 분리해 앞에 tsu0만 가져옴.
                i += 1
                if i >= n:
                    break


def img_generating(main_dir, categories, n=3):
    """ 폴더(class별로 나누어져 저장) 안의 이미지들을 불러와 이미지를 n장씩 더 부풀림
    :param main_dir: class별로 저장되어 있는 폴더들의 상위폴더 path를 기입
    :param categories: [] 리스트 형태로 기입
    :param n: n장만큼 부풀림
    :return: 해당폴더에 n장만큼 부풀려 저장
    """
    datagen = ImageDataGenerator(rescale=1. / 255,
                                 rotation_range=30,  # 각도 범위내 회전
                                 width_shift_range=0.1,  # 수평방향
                                 height_shift_range=0.1,  # 수직방향
                                 brightness_range=[0.2, 1.5],  # 밝기
                                 shear_range=0.7,  # 시계반대방향
                                 zoom_range=[0.8, 1.0],
                                 horizontal_flip=True,  # 수평방향 뒤집기
                                 vertical_flip=True,
                                 fill_mode='nearest')

    for cat in categories:
        cat_dir = main_dir + cat   # 각 카테고리별 디렉토리
        files = glob.glob(cat_dir + "/*.jpg")   # cat_dir의 jpg 파일들 다 가져옴

        for file in files:
            file_size = os.stat(file)
            size = file_size.st_size / 1024  # KB로 변환

            img = load_img(file)
            if size >= 256:   # 이미지 크기가 256 KB 이상이면 이미지 픽셀을 (64, 64)로 resize한 뒤 복제
                img = img.convert('RGB').resize((256, 256), Image.ANTIALIAS)   # Image.ANTIALIAS는 resizing을 부드럽게 처리하기 위함)

            x = img_to_array(img)
            x = x.reshape((1,) + x.shape)
            # x 전체를 하나의 []로 묶음 -> 뒤에서 Conv2D할 때 input이 4차원형태로 들어가야 되기 때문에 미리 변환해놓음
            i = 0
            for _ in datagen.flow(x, batch_size=1, save_to_dir=cat_dir,
                                  save_prefix=os.path.basename(file).split('.')[0]+'_copy', # file에서 맨마지막 부분인 tsu0.jpg를 출력하고, '.'으로 분리해 앞에 tsu0만 가져옴.
                                  save_format='jpg'):
                i += 1
                if i >= n:
                    break

            # # 위 코드 보충설명  (https://devanix.tistory.com/298 참고)
            # file_path = "./practice3/tsu/tsu0.jpg"
            # x = os.path.basename(file_path).split('.')[0]
            # file에서 맨마지막 부분인 tsu0.jpg를 출력하고, '.'으로 분리해 앞에 tsu0만 가져옴.
            # print(x)  # tsu0 출력됨.


def img_to_dataset(main_dir, categories, size=(64,64)):
    """ 폴더 안의 파일을 불러와 픽셀을 배열로 변환하여 저장 """
    X = []
    for cat in categories:
        cat_dir = main_dir + cat
        files = glob.glob(cat_dir + "/*.jpg")
        for file in files:
            im = Image.open(file)
            im = im.convert('RGB').resize(size)  # rgb로 변환해야 jpg로 변환 가능함
            im = img_to_array(im)   # list
            X.append(im)
    return np.asarray(X)  # list를 asarray로 변환


def labeling(main_dir, categories):
    """ 상위폴더 안에있는 폴더(categories)들 안에 이미지를
        one-hot-encoding 하여 array로 저장 """
    y = []
    cat_num = len(categories)
    for idx, cat in enumerate(categories):
        cat_dir = main_dir + cat
        files = glob.glob(cat_dir + "/*.jpg")  # 'ak'폴더 안에 있는 이미지 파일의 이름을 불러옴.
        for file in files:
            # if f'{cat}' in f'{file}':  # 이건 파일이름을 폴더이름과 같게 변경해야하는 단점이 있음.
            if os.path.isfile(file):
                label = [0 for i in range(cat_num)]
                label[idx] = 1
                y.append(label)
    return np.array(y)


def number_of_files(main_dir, categories):
    """ 상위폴더 안 하위폴더에 들어있는 파일의 갯수 각각 세어주고,
        총합을 반환"""
    n_list = []
    for cat in categories:
        cat_dir = main_dir + cat
        files = glob.glob(cat_dir + "/*.jpg")
        # 폴더별로 곱할 수 저장
        # print(f'{cat}폴더의 파일 갯수: {len(files)}')
        n_list.append(len(files))
    return n_list


def mean_of_img_size(main_dir, categories):
    """ 폴더별 이미지 파일들의 크기 평균을 KB로 반환 """
    file_size = []
    for cat in categories:
        cat_dir = main_dir + cat
        files = glob.glob(cat_dir + "/*.jpg")
        # print(files)
        for file in files:
            size = os.stat(file)  # 각 이미지 파일의 사이즈 알아봄.
            file_size.append(size.st_size / 1024)  # KB로 변환하여 리스트로 저장
        print(f'{cat}폴더의 이미지 크기 평균: {np.mean(file_size)} KB')


if __name__ == '__main__':
    # 이미지 부풀리기
    main_dir = './img_class/'
    categories = ["ak", "bcc", "df", "mel", "nv", "pbk", "scc", "sk", "tsu", "vl"]
    # img_generating(main_dir=main_dir, categories=categories, n=3)
    img_generating_2(main_dir, categories, n=3)


    # X 데이터셋 만들기 (class별로 나누어져 있는 폴더에서 이미지를 가져와 픽셀 배열로 저장)
    X = img_to_dataset(main_dir=main_dir, categories=categories)
    print(X.shape)
    print(len(X))

    # 폴더 안에있는 이미지에 y label 붙여 array로 저장
    y = labeling(main_dir=main_dir, categories=categories)
    print(y[:10])
    print(len(y))


    # 디렉토리 내의 파일 개수 알아보기
    main_dir = './img_class/'
    categories = ["df", "mel", "nv", "tsu", "vl"]   # 육안으로 보았을 때 쯔쯔가무시와 비슷해보이는 5개 클래스를 선정
    number_of_files = number_of_files(main_dir=main_dir, categories=categories)
    print(number_of_files)


    # 이미지 파일들의 크기 평균 알아보기
    mean_of_img_size(main_dir=main_dir, categories=categories)




