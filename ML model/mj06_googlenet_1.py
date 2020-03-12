""" googlenet 2.18 화 ~
 - 각 클래스당 이미지를 2,000장 정도로 부풀려 클래스당 비율을 맞춤
 - 이미지 제너레이터는 보통수준으로 함(많이 변형하지 않았음)
 - 이미지 파일크기가 250KB 보다 큰 경우, 픽셀 256 비율에 맞게 축소하고,
   작은 경우, 픽셀 224 비율에 맞게 축소하여 부풀림.
 - 모델 넣어주기 전 (224,224,3)로 resize하여 array로 저장(4기가 이상은 protocol=4로 변경하니 됨..총6.4기가)
"""

from tensorflow.keras.models import Model
from keras.models import Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, GlobalAveragePooling2D, Dense, AveragePooling2D, Flatten , \
    Concatenate, BatchNormalization
from tensorflow.keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
import os
import numpy as np
import matplotlib.pyplot as plt

from keras.engine.saving import load_model
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from image_preporcessing.mj03_2_img_generating_upgrade import img_to_dataset, labeling, \
    number_of_files, img_generating_2


def inception(input_tensor, filter_channels):
    """ Inception 모듈 정의
    :param input_tensor: (h,w,n), h와 w는 같은 값이여야함 """
    # 64,(96,128),(16,32),32
    # 출력 피쳐맵의 갯수가 64, 두번째 1x1, 3x3 convolution 연산의 출력 피쳐맵의 갯수는 각각 96,128,
    # 세번째 1x1, 5x5 convolution 연산의 출력 피쳐맵의 갯수는 각각 16,32
    # 마지막 maxpooling 후의 1x1 convolution 연산의 출력 피쳐맵의 갯수는 32 란 의미 이다.
    filter_1x1, filter_3x3_R, filter_3x3, filter_5x5_R, filter_5x5, pool_proj = filter_channels

    branch_1 = Conv2D(filters=filter_1x1, kernel_size=(1, 1), strides=1, padding='same',
                      activation='relu', kernel_initializer='he_normal')(input_tensor)  # He 정규분포 초기값 설정

    branch_2 = Conv2D(filter_3x3_R, (1, 1), strides=1, padding='same',
                      activation='relu', kernel_initializer='he_normal')(input_tensor)
    branch_2 = Conv2D(filter_3x3, (3, 3), strides=1, padding='same',
                      activation='relu', kernel_initializer='he_normal')(branch_2)

    branch_3 = Conv2D(filter_5x5_R, (1, 1), strides=1, padding='same', activation='relu',
                      kernel_initializer='he_normal')(input_tensor)
    branch_3 = Conv2D(filter_5x5, (5, 5), strides=1, padding='same', activation='relu',
                      kernel_initializer='he_normal')(branch_3)

    branch_4 = MaxPooling2D((3, 3), strides=1, padding='same')(input_tensor)
    branch_4 = Conv2D(pool_proj, (1, 1), strides=1, padding='same', activation='relu',
                      kernel_initializer='he_normal')(branch_4)

    DepthConcat = Concatenate()([branch_1, branch_2, branch_3, branch_4])

    return DepthConcat


def GoogLeNet(model_input, classes=5):
    # classes=10에서 5로 변경, 5개 중에서 분류할 것이기 때문..
    conv_1 = Conv2D(64, (7, 7), strides=2, padding='same', activation='relu')(model_input) # (112, 112, 64)
    pool_1 = MaxPooling2D((3, 3), strides=2, padding='same')(conv_1) # (56, 56, 64)
    # LRN_1 = LocalResponseNormalization()(pool_1) # (56, 56, 64) 이걸 BatchNormalization으로 대체함
    BN_1 = BatchNormalization()(pool_1)

    conv_2 = Conv2D(64, (1, 1), strides=1, padding='valid', activation='relu')(BN_1) # (56, 56, 64)
    conv_3 = Conv2D(192, (3, 3), strides=1, padding='same', activation='relu')(conv_2) # (56, 56, 192)
    BN_2 = BatchNormalization()(conv_3) # (56, 56, 192)
    pool_2 = MaxPooling2D((3, 3), strides=2, padding='same')(BN_2) # (28, 28, 192)

    inception_3a = inception(pool_2, [64, 96, 128, 16, 32, 32])  # (28, 28, 256)
    inception_3b = inception(inception_3a, [128, 128, 192, 32, 96, 64])  # (28, 28, 480)

    pool_3 = MaxPooling2D((3, 3), strides=2, padding='same')(inception_3b)  # (14, 14, 480)

    inception_4a = inception(pool_3, [192, 96, 208, 16, 48, 64]) # (14, 14, 512)
    inception_4b = inception(inception_4a, [160, 112, 224, 24, 64, 64]) # (14, 14, 512)
    inception_4c = inception(inception_4b, [128, 128, 256, 24, 64, 64]) # (14, 14, 512)
    inception_4d = inception(inception_4c, [112, 144, 288, 32, 64, 64]) # (14, 14, 528)
    inception_4e = inception(inception_4d, [256, 160, 320, 32, 128, 128]) # (14, 14, 832)

    pool_4 = MaxPooling2D((3, 3), strides=2, padding='same')(inception_4e)  # (7, 7, 832)

    inception_5a = inception(pool_4, [256, 160, 320, 32, 128, 128])  # (7, 7, 832)
    inception_5b = inception(inception_5a, [384, 192, 384, 48, 128, 128])  # (7, 7, 1024)

    avg_pool = GlobalAveragePooling2D()(inception_5b)
    dropout_1 = Dropout(rate=0.4)(avg_pool)

    linear = Dense(1000, activation='relu')(dropout_1)  # 펴줄때 뉴런수를 말하는듯..? unit을 1000에서 classes(=5)로 바꿈..

    model_output = Dense(classes, activation='softmax', name='main_classifier')(linear)  # 'softmax'
    # classes=5로 설정해줌


    # Auxiliary Classifier
    auxiliary_4a = AveragePooling2D((5, 5), strides=3, padding='valid')(inception_4a)
    auxiliary_4a = Conv2D(128, (1, 1), strides=1, padding='same', activation='relu')(auxiliary_4a)
    auxiliary_4a = Flatten()(auxiliary_4a)
    auxiliary_4a = Dense(1024, activation='relu')(auxiliary_4a)
    auxiliary_4a = Dropout(rate=0.7)(auxiliary_4a)
    auxiliary_4a = Dense(classes, activation='softmax', name='auxiliary_4a')(auxiliary_4a)

    auxiliary_4d = AveragePooling2D((5, 5), strides=3, padding='valid')(inception_4d)
    auxiliary_4d = Conv2D(128, (1, 1), strides=1, padding='same', activation='relu')(auxiliary_4d)
    auxiliary_4d = Flatten()(auxiliary_4d)
    auxiliary_4d = Dense(1024, activation='relu')(auxiliary_4d)
    auxiliary_4d = Dropout(rate=0.7)(auxiliary_4d)
    auxiliary_4d = Dense(classes, activation='softmax', name='auxiliary_4d')(auxiliary_4d)

    model = Model(model_input, [model_output, auxiliary_4a, auxiliary_4d])

    return model



if __name__ == '__main__':
    # 디렉토리 내의 파일 개수 알아보기
    main_dir = './img_class_2/'
    categories = ["df", "mel", "nv", "tsu", "vl"]
    number_of_files = number_of_files(main_dir=main_dir, categories=categories)
    print(number_of_files)

    k = 2000   # 부풀릴 이미지 목표 장수
    n = [k/i for i in number_of_files]
    print(f'폴더별로 곱할 수: {n}')


    # 원본 이미지 불러와 부풀리기
    categories = ["df"]
    img_generating_2(main_dir=main_dir, categories=categories, n=18)

    categories = ["mel"]
    img_generating_2(main_dir=main_dir, categories=categories, n=4)

    categories = ["nv"]
    img_generating_2(main_dir=main_dir, categories=categories, n=5)

    categories = ["tsu"]
    img_generating_2(main_dir=main_dir, categories=categories, n=3)

    categories = ["vl"]
    img_generating_2(main_dir=main_dir, categories=categories, n=14)


    # X 데이터셋 만들기 (class별로 나누어져 있는 폴더에서 이미지를 가져와 픽셀 배열로 저장)
    X = img_to_dataset(main_dir=main_dir, categories=categories, size=(224,224))
    # 부풀리기 할때는 비율에 맞게 부풀리고, 배열로 만들 때, 224로 resize해서 GoogLeNet에 넣음.
    print(X.shape)   # (11431, 224, 224, 3)
    print(len(X))

    # 폴더 안에있는 이미지에 y label 붙여 array로 저장
    y = labeling(main_dir=main_dir, categories=categories)
    print(y[:10])
    print(len(y))


    # train, test 데이터로 나누기
    # np.random.seed(219)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    print(len(X_train), len(X_test), len(y_train), len(y_test))
    print('X_train:', X_train[0:2])
    print('y_train:', y_train[0:2])
    print('X_test', X_test[0:2])
    print('y_test', y_test[0:2])

    xy = (X_train, X_test, y_train, y_test)
    np.save('./xy_googlenet_1_save.npy', xy)   # x_save.npy
    # 변환 용량이 4기가 이상이면 메모리 오류떠서 아래 프로토콜을 3에서 4로 바꾸고 변환했더니 됨..
    # format 파일 -> pickle.dump(array, fp, protocol=4, **pickle_kwargs)


    ### np.save로 저장된 파일 불러오기
    X_train, X_test, y_train, y_test = np.load('./xy_googlenet_1_save.npy', allow_pickle=True)

    # 정규화 시키기
    X_train = X_train.astype('float16') / 255
    X_test = X_test.astype('float16') / 255

    print(f'X_train: {X_train.shape}, y_train: {y_train.shape}')  # X_train: (9144, 224, 224, 3), y_train: (9144, 5)
    print(f'X_test: {X_test.shape}, y_test: {y_test.shape}')  # X_test: (2287, 224, 224, 3), y_test: (2287, 5)

    print(f'X_train: {X_train[0:2]}, X_test: {X_test[0:2]}')


    ## 업그레이드할 때, 정규화 시켜서 저장해놓기 -> 시간이 너무 오래걸림..


# ############### 참고용
    # input_shape = (224, 224, 3)
    # (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    #
    # print(x_train)
    # print(x_train[0].shape)  # (32, 32, 3)
    # print(x_train.shape)  # (50000, 32, 32, 3)
    # print(type(x_train))  # <class 'numpy.ndarray'>
    #
    # # 이미지 데이터를 224로 resize 해주는 과정
    # x_train = Upscaling_Data(x_train, input_shape)
    # x_test = Upscaling_Data(x_test, input_shape)

    # x_train = np.float32(x_train / 255.)
    # x_test = np.float32(x_test / 255.)
    #
    # y_train = to_categorical(y_train, num_classes=10)
    # y_test = to_categorical(y_test, num_classes=10)




    ## GoogLeNet 모델 만들기 ###

    input_shape = (224, 224, 3)  # 이 사이즈로 미리 변환해두었음.

    # model = Sequential()
    # keras.applications.inception_v3.InceptionV3(include_top=True, weights='imagenet', input_tensor=None,
    #                                             input_shape=True, pooling=None, classes=5)

    model_input = Input(shape=input_shape)
    model = GoogLeNet(model_input=model_input, classes=5)  # class 5개에서 분류

    optimizer = SGD(momentum=0.9)

    # 컴파일
    model.compile(optimizer, loss={'main_classifier': 'categorical_crossentropy',
                                   'auxiliary_4a': 'categorical_crossentropy',
                                   'auxiliary_4d': 'categorical_crossentropy'},
                  loss_weights={'main_classifier': 1.0,
                                'auxiliary_4a': 0.3,   # 중간에 기울기 소실 문제 해결을 위해 soft함수 뽑아냄. loss 30%만 참고(?)
                                'auxiliary_4d': 0.3},
                  metrics=['accuracy'])

    # 계층 요약 확인
    model.summary()
    # Total params: 11,344,559
    # Trainable params: 11,344,047
    # Non-trainable params: 512



    # 신경망 모델의 성능 향상이 없는 경우 중간에 epoch을 빨리 중지시키기 위해서
    early_stop = EarlyStopping(monitor='val_loss',
                               verbose=1,
                               patience=10)

    # 신경망 학습모델 파일로 저장
    model_dir = "./model"
    if not os.path.exists(model_dir):  # model_dir이 없을 경우 폴더 생성
        os.mkdir(model_dir)

    model_path = model_dir + '/googlenet_1.model'
    checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=1, save_best_only=True)
    early_stop = EarlyStopping(monitor='val_loss', patience=5)

    # 신경망 학습
    history = model.fit(X_train, {'main_classifier': y_train,
                                  'auxiliary_4a': y_train,
                                  'auxiliary_4d': y_train},
                        epochs=50,     # 에폭만큼 파라미터 업데이트
                        batch_size=100,  # 전체갯수를 batch_size로 나눈 만큼 반복
                        callbacks=[checkpoint, early_stop],
                        validation_split=0.2)


    # 테스트 데이터를 사용해서 신경망 모델을 평가
    # 테스트 데이터의 Loss, Accuracy
    eval = model.evaluate(X_test, y_test)
    print(f'Test loss: {eval[0]}, accuracy: {eval[1]}')

    # loss: 0.3052 - main_classifier_loss: 0.1919 - auxiliary_4a_loss: 0.1928 - auxiliary_4d_loss: 0.1924
    # main_classifier_accuracy: 0.9224 - auxiliary_4a_accuracy: 0.9260 - auxiliary_4d_accuracy: 0.9218
    # val_loss: 0.5202 - val_main_classifier_loss: 0.3203 - val_auxiliary_4a_loss: 0.3194 - val_auxiliary_4d_loss: 0.3255
    # val_main_classifier_accuracy: 0.8808 - val_auxiliary_4a_accuracy: 0.8721 - val_auxiliary_4d_accuracy: 0.8775



    # 학습 데이터와 테스트 데이터의 Loss 그래프
    train_loss = history.history['loss']  # history dictionary에 저장된 'loss' 키를 갖는 value들을 가져옴
    val_loss = history.history['val_loss']

    x = range(len(train_loss))
    plt.plot(x, train_loss, marker='.', color='red', label='Train loss')
    plt.plot(x, val_loss, marker='.', color='blue', label='Val loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.show()

    # 학습 데이터, 테스트 데이터의 정확도 그래프
    train_accuracy = history.history['acc']
    val_accuracy = history.history['val_acc']

    x = range(len(train_accuracy))
    plt.plot(x, train_accuracy, marker='.', color='red', label='Train loss')
    plt.plot(x, val_accuracy, marker='.', color='blue', label='Val loss')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()




    # 저장한 학습모델 불러옴.
    model = load_model('./model/googlenet_1.model')

    # confusion matrix & classification report
    print(y_test)

    y_true = np.argmax(y_test, axis=1)   # 행 중 가장 큰값의 idx
    # (one-hot-encoding 되어있으므로 1로 표시된 값을 행에서 가장 큰값으로 출력하여 array로 만듦)
    print(y_true)
    print(X_test[0])
    y_pred = np.argmax(model.predict(X_test), axis=1)
    print(y_pred)

    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    # 세로가 실제클래스, 가로가 예측클래스
    ## 1 - 왜 이렇게 3줄만 나오는지 모르겠음....
    # [[1504    2    0    1    0]
    #  [   0    2    0    1  937]
    #  [1758    1    0   26    0]]


    report = classification_report(y_true, y_pred, target_names=categories)
    print(report)
    ## 1


