"""
- xy_googlenet_1_save.npy = 2,000장씩 (224,224,3)
"""

from keras.applications.inception_v3 import InceptionV3
from image_preporcessing.mj03_2_img_generating_upgrade import img_to_dataset, labeling, \
    number_of_files, img_generating_2
import numpy as np
import os
import matplotlib.pyplot as plt
from keras.models import Model
from keras.models import Input
from keras.layers import Conv2D, MaxPooling2D, Dropout, GlobalAveragePooling2D, Dense, AveragePooling2D, Flatten , \
    Concatenate, BatchNormalization
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback

from keras.engine.saving import load_model
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    categories = ["df", "mel", "nv", "tsu", "vl"]


    # ### np.save로 저장된 파일 불러오기
    # X_train, X_test, y_train, y_test = np.load('./xy_googlenet_1_save.npy', allow_pickle=True)
    # # xy5_cnn_final_3_save.npy: 5,000장씩 (128,128)
    # # xy_googlenet_1_save.npy: 2,000장씩 (224,224)
    #
    #
    #
    #
    # pre_trained model 불러오기
    input_shape = (224, 224, 3)
    base_model = InceptionV3(include_top=False,
                             weights='imagenet',
                             input_shape=input_shape)
    x = base_model.output
    x = Dropout(0.5)(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    predictions = Dense(5, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # 컴파일
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.summary()



    # 신경망 모델의 성능 향상이 없는 경우 중간에 epoch을 빨리 중지시키기 위해서
    early_stop = EarlyStopping(monitor='val_loss',
                               verbose=1,
                               patience=10)

    # 신경망 학습모델 파일로 저장
    model_dir = "./model"
    if not os.path.exists(model_dir):  # model_dir이 없을 경우 폴더 생성
        os.mkdir(model_dir)

    model_path = model_dir + '/Inception_v3_1.model'
    checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=1, save_best_only=True)
    early_stop = EarlyStopping(monitor='val_loss', patience=5)

    # 신경망 학습
    history = model.fit(X_train, y_train,
                        epochs=50,     # 에폭만큼 파라미터 업데이트
                        batch_size=400,  # 전체갯수를 batch_size로 나눈 만큼 반복
                        callbacks=[checkpoint, early_stop],
                        validation_split=0.2)



    # 테스트 데이터를 사용해서 신경망 모델을 평가
    # 테스트 데이터의 Loss, Accuracy
    eval = model.evaluate(X_test, y_test)
    print(f'Test loss: {eval[0]}, accuracy: {eval[1]}')


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

    #
    #
    #
    #
    # # 저장한 학습모델 불러옴.
    # model = load_model('./model/Inception_v3_1.model')
    #
    # # confusion matrix & classification report
    # print(y_test)
    #
    # y_true = np.argmax(y_test, axis=1)   # 행 중 가장 큰값의 idx
    # # (one-hot-encoding 되어있으므로 1로 표시된 값을 행에서 가장 큰값으로 출력하여 array로 만듦)
    # print(y_true)
    # print(X_test[0])
    # y_pred = np.argmax(model.predict(X_test), axis=1)
    # print(y_pred)
    #
    # cm = confusion_matrix(y_true, y_pred)
    # print(cm)
    # # 세로가 실제클래스, 가로가 예측클래스
    #
    #
    #
    # report = classification_report(y_true, y_pred, target_names=categories)
    # print(report)
    # ## 1
