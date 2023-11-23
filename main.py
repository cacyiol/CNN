import os
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json
def load_and_preprocess_data():
    # 读取训练集数据
    train = pd.read_csv('Train/train.csv')

    # 创建训练图像路径列表
    train_image_paths = ['Train/train/' + img_name for img_name in train['id'].astype('str') + '.png']

    # 创建训练标签
    y = to_categorical(train['label'].values)

    # 加载和预处理训练图像
    train_image = []
    for img_path in tqdm(train_image_paths):
        img = image.load_img(img_path, target_size=(28, 28, 3))
        img = image.img_to_array(img)
        img = img / 255
        train_image.append(img)

    # 将训练图像转换为 numpy 数组
    X = np.array(train_image)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

    return X_train, y_train, X_test, y_test

def build_and_train_model(X_train, y_train, X_test, y_test):
    # 构建模型
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 3)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(y_train.shape[1], activation='softmax'))

    # 编译模型
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

    # 添加 ModelCheckpoint 回调函数用于保存效果最好的模型
    checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1, patience=3)

    # 训练模型
    model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), callbacks=[checkpoint])

    # 计算最终准确率
    _, accuracy = model.evaluate(X_test, y_test)
    print(f'Final Accuracy: {accuracy * 100:.2f}%')

    return model

def save_best_model(model, filename='best_model.h5'):
    model.save(filename)
    print(f'Model saved as {filename}')

def load_and_preprocess_test_data():
    # 读取测试集数据
    test = pd.read_csv('Test/test.csv')

    # 创建测试图像路径列表
    test_image_paths = ['Test/test/' + img_name for img_name in test['id'].astype('str') + '.png']

    # 加载和预处理测试图像
    test_image = []
    for img_path in tqdm(test_image_paths):
        img = image.load_img(img_path, target_size=(28, 28, 3))
        img = image.img_to_array(img)
        img = img / 255
        test_image.append(img)

    # 将测试图像转换为 numpy 数组
    test_data = np.array(test_image)

    return test_data



def predict_single_image(model, img_path, unknown_threshold=0.5):
    img = image.load_img(img_path, target_size=(28, 28, 3))
    img_array = image.img_to_array(img)
    img_array = img_array / 255
    img_array = np.expand_dims(img_array, axis=0)  # 添加批次维度

    # 进行预测
    prediction = model.predict(img_array)
    predicted_label = np.argmax(prediction)

    # 获取对应标签的概率
    predicted_probabilities = prediction[0]

    # 判断是否为未知
    if all(prob < unknown_threshold for prob in predicted_probabilities):
        predicted_label = -1  # -1 表示未知

    return predicted_label

def train():
    # 加载和预处理训练数据
    X_train, y_train, X_test, y_test = load_and_preprocess_data()

    # 构建并训练模型
    best_model = build_and_train_model(X_train, y_train, X_test, y_test)

    # 保存效果最好的模型
    save_best_model(best_model)

def test():
    # 加载已保存的最好模型
    best_model = load_model('best_model.h5')

    # 加载和预处理测试数据
    test_data = load_and_preprocess_test_data()

    # 进行预测
    predictions = np.argmax(best_model.predict(test_data), axis=1)
    print(f'Predictions for all test images: {predictions}')




def testSingle():
    # 加载最佳模型
    best_model = load_model('best_model.h5')

    # 选择一张测试图像进行单张预测
    # test_img_path = 'samplePhotos/waitao.png'  # 替换成实际的图像路径
    test_img_path = 'samplePhotos/none.png' # 无关图片测试
    predicted_label = predict_single_image(best_model, test_img_path)

    # 加载标签映射文件
    with open('label_mapping.json', 'r', encoding='utf-8') as f:
        label_mapping = json.load(f)

    # 获取预测标签对应的服饰名
    predicted_category = label_mapping.get(str(predicted_label), '未知')

    print(f'The predicted category for the example image is: {predicted_category}')




def main():
    # train()  # 训练模型
    # test()   # 测试模型
    testSingle()   # 预测单张图片

if __name__ == "__main__":
    main()

