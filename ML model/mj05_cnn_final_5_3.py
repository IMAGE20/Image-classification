""" 5개 class를 cnn으로 분류하여 알아맞히기
 - 각 클래스당 이미지를 5,000장 정도로 부풀려 클래스당 비율을 맞춤
 - 이미지 제너레이터는 보통수준으로 함(많이 변형하지 않았음)
 - 이미지 파일크기가 250KB 보다 큰 경우, 픽셀 256 비율에 맞게 축소하고,
   작은 경우, 픽셀 128 비율에 맞게 축소하여 부풀림.
   이미지에 노이즈를 즐여주도록 했음.
 - cnn 모델은 conv2D 4개층으로 함.
"""
from keras.engine.saving import load_model
from sklearn.metrics import confusion_matrix , classification_report
from sklearn.model_selection import train_test_split
import glob
from keras_preprocessing.image import ImageDataGenerator , load_img , img_to_array
from keras import Sequential
from keras.callbacks import EarlyStopping , ModelCheckpoint
from keras.layers import Conv2D , MaxPool2D , Flatten , Dense , Dropout
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from PIL import Image
import os
from image_preporcessing.mj03_2_img_generating_upgrade import img_to_dataset , labeling , \
    number_of_files , img_generating_2
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # 디렉토리 내의 파일 개수 알아보기
    main_dir = './img_class/'
    categories = ["df", "mel", "nv", "tsu", "vl"]   # 육안으로 보았을 때 쯔쯔가무시와 비슷해보이는 5개 클래스를 선정
    # number_of_files = number_of_files(main_dir=main_dir, categories=categories)
    # print(number_of_files)
    #
    # k = 2000   # 부풀릴 이미지 목표 장수
    # n = [k/i for i in number_of_files]
    # print(f'폴더별로 곱할 수: {n}')


    # # 원본 이미지 불러와 부풀리기
    # categories = ["df"]
    # img_generating_2(main_dir=main_dir, categories=categories, n=0.9)
    #
    # categories = ["mel"]
    # img_generating_2(main_dir=main_dir, categories=categories, n=0.9)
    #
    # categories = ["nv"]
    # img_generating_2(main_dir=main_dir, categories=categories, n=0.9)
    #
    # categories = ["tsu"]
    # img_generating_2(main_dir=main_dir, categories=categories, n=0.75)
    #
    # categories = ["vl"]
    # img_generating_2(main_dir=main_dir, categories=categories, n=0.9)


    # # X 데이터셋 만들기 (class별로 나누어져 있는 폴더에서 이미지를 가져와 픽셀 배열로 저장)
    # X = img_to_dataset(main_dir=main_dir, categories=categories, size=(128,128))
    # print(X.shape)   # (22861, 128, 128, 3)
    # print(len(X))
    #
    # # 폴더 안에있는 이미지에 y label 붙여 array로 저장
    # y = labeling(main_dir=main_dir, categories=categories)
    # print(y[:10])
    # print(len(y))
    #
    #
    # # train, test 데이터로 나누고, 데이터 정규화 하기
    # np.random.seed(214)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # print(len(X_train), len(X_test), len(y_train), len(y_test))
    # print('X_train:', X_train[0:2])
    # print('y_train:', y_train[0:2])
    # print('X_test', X_test[0:2])
    # print('y_test', y_test[0:2])

    xy = (X_train, X_test, y_train, y_test)
    np.save('./xy5_cnn_final_3_save.npy', xy)   # x_save.npy


    ### np.save로 저장한 내용 불러오기
    X_train, X_test, y_train, y_test = np.load('./xy5_cnn_final_3_save.npy', allow_pickle=True)
    print(X_train.shape)

    # # 정규화 시키기
    # X_train = X_train.astype('float16') / 255
    # X_test = X_test.astype('float16') / 255
    #
    # print(f'X_train: {X_train.shape}, y_train: {y_train.shape}')
    # print(f'X_test: {X_test.shape}, y_test: {y_test.shape}')
    #
    # print(f'X_train: {X_train[0:2]}, X_test: {X_test[0:2]}')



    ### CNN 모델 만들기 ###
    # 신경망 모델 생성 - Sequential 클래스 인스턴스 생성
    model = Sequential()

    # 신경망 모델에 은닉층, 출력층 계층(layers)들을 추가
    # (Conv2D -> MaxPool2D) x 4개층 -> Flatten -> Dense -> Dense
    # Conv2D 활성화 함수: ReLU
    # Dense 활성화 함수: ReLU, Softmax
    model.add(Conv2D(filters=32,         # 필터 갯수
                     kernel_size=(3,3),  # 필터의 height/width
                     activation='relu',  # 활성화 함수
                     input_shape=(128, 128, 3),
                     padding='same'))  # 입력데이터의 shape (h,w,c)순서임.
    model.add(MaxPool2D(pool_size=2))   # 이미지가 줄어듦.
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=32,
                     kernel_size=(3,3),
                     activation='relu',
                     padding='same'))
    model.add(MaxPool2D(pool_size=2))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=32,
                     kernel_size=(3,3),
                     activation='relu',
                     padding='same'))
    model.add(MaxPool2D(pool_size=2))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=32,
                     kernel_size=(3,3),
                     activation='relu',
                     padding='same'))
    model.add(MaxPool2D(pool_size=2))
    model.add(Dropout(0.25))

    model.add(Flatten())  # keras에서는 Dense 층에 넣기 전에 모두 펴줘야함.
    model.add(Dense(128, activation='relu'))   # 완전 연결 은닉층
    model.add(Dense(len(categories), activation='softmax'))  # 출력층 (위에서 함수로 one-hot-encoding 해줌)


    model.summary()

    # 신경망 모델 컴파일
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # 신경망 모델의 성능 향상이 없는 경우 중간에 epoch을 빨리 중지시키기 위해서
    early_stop = EarlyStopping(monitor='val_loss',
                               verbose=1,
                               patience=10)

    # 신경망 학습
    model_dir = "./model"
    if not os.path.exists(model_dir):  # model_dir이 없을 경우 폴더 생성
        os.mkdir(model_dir)

    model_path = model_dir + '/cnn_final_3.model'
    checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=1, save_best_only=True)
    early_stop = EarlyStopping(monitor='val_loss', patience=10)
    model.summary()

    history = model.fit(X_train, y_train,
                        batch_size=200,  # 전체갯수를 batch_size로 나눈 만큼 반복
                        epochs=100,   # 에폭만큼 파라미터 업데이트
                        verbose=1,
                        callbacks=[checkpoint, early_stop],
                        validation_split=0.2)


    # 테스트 데이터를 사용해서 신경망 모델을 평가
    # 테스트 데이터의 Loss, Accuracy
    eval = model.evaluate(X_test, y_test)
    print(f'Test loss: {eval[0]}, accuracy: {eval[1]}')
    # - loss: 0.1903 - accuracy: 0.9273 - val_loss: 0.5703 - val_accuracy: 0.8141
    # Test loss: 0.5832964925720121, accuracy: 0.8250983953475952
    # => 정확도 낮은 수준이고, 오버피팅도 심함..


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
    train_accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    x = range(len(train_accuracy))
    plt.plot(x, train_accuracy, marker='.', color='red', label='Train loss')
    plt.plot(x, val_accuracy, marker='.', color='blue', label='Val loss')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()


    # # 저장한 학습모델 불러옴.
    #
    #
    # # confusion matrix & classification report
    # print(y_test)
    # y_true = np.argmax(y_test, axis=1)   # 행 중 가장 큰값의 idx
    # # (one-hot-encoding 되어있으므로 1로 표시된 값을 행에서 가장 큰값으로 출력하여 array로 만듦)
    # print(y_true)
    # y_pred = np.argmax(model.predict(X_test), axis=1)
    # print(y_pred)
    #
    # cm = confusion_matrix(y_true, y_pred)
    # print(cm)
    # # 세로가 실제클래스, 가로가 예측클래스
    ### 1
    # # [[336  20  43   7   6]
    # #  [ 14 331  55  26   6]
    # #  [ 42 127 285   3   8]
    # #  [ 13  23   7 477  24]
    # #  [  4   3   3  13 411]]
    #
    #
    # report = classification_report(y_true, y_pred, target_names=categories)
    # print(report)
    ### 1
    # #               precision    recall  f1-score   support
    # #
    # #           df       0.82      0.82      0.82       412
    # #          mel       0.66      0.77      0.71       432
    # #           nv       0.73      0.61      0.66       465
    # #          tsu       0.91      0.88      0.89       544
    # #           vl       0.90      0.95      0.92       434
    # #
    # #     accuracy                           0.80      2287
    # #    macro avg       0.80      0.80      0.80      2287
    # # weighted avg       0.81      0.80      0.80      2287




