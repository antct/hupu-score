import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
import numpy as np

from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from PIL import Image
from tensorflow.keras.optimizers import Adam

import os
os.environ['CUDA_VISIBLE_DEVICES']='0'  
 
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--mode', dest='mode', default='fine_tune')
parser.add_argument('--input_size', dest='input_size', default=256)
parser.add_argument('--batch_size', dest='batch_size', default=32)
parser.add_argument('--fix_layers', dest='fix_layers', default=0)
parser.add_argument('--max_epochs', dest='max_epochs', default=100)
parser.add_argument('--patience', dest='patience', default=8)
parser.add_argument('--image', dest='image')
args = parser.parse_args()


def r_square(y_true, y_pred):
    SSR = K.sum(K.square(y_pred-y_true), axis=-1)
    SST = K.sum(K.square(y_true-K.mean(y_true)), axis=-1)
    return 1 - SSR/(SST+1e-6)


class model():
    def __init__(self, args):
        self.model = None
        self.args = args


    def _build_SCUT_train_data(self):
        data = []
        with open('./SCUT/label.txt', 'r') as f:
            data = f.readlines()
        data = [i.strip('\n').split(' ') for i in data]
        images = []
        labels = []
        for i in data:
            if i[0][:2] != 'AF':
                continue
            try:
                fname =  './face/SCUT/{}'.format(i[0])
                images.append(np.array(Image.open(fname).convert('RGB').resize((self.args.input_size, self.args.input_size), Image.LANCZOS)) / 255.0)
                labels.append(float(i[1])*2-1)
            except:
                continue

        images = np.array(images)
        labels = np.array(labels)
        return images, labels

    def _build_hupu_train_data(self):
        scores = {}
        with open('./hupu/score.json', 'r') as f:
            scores = dict(eval(f.read()))
        data = []
        for i in os.listdir('./face/hupu'):
            if str(i).endswith('jpg'):
                data.append(i)

        images = []
        labels = []        
        for i in data:
            fname = './face/hupu/{}'.format(i)
            images.append(np.array(Image.open(fname).convert('RGB').resize((self.args.input_size, self.args.input_size), Image.LANCZOS)) / 255.0)
            score = float(scores[i[:-4]])
            labels.append(score)

        avg_score = sum(labels)/len(labels)
        labels = [avg_score + (i - avg_score) * 1.25 for i in labels]

        images = images
        labels = labels

        images = np.array(images)
        labels = np.array(labels)
        return images, labels
    
    def _build_train_data(self):
        hupu_images, hupu_labels = self._build_hupu_train_data()
        SCUT_images, SCUT_labels = self._build_SCUT_train_data()
        images = np.concatenate((hupu_images, SCUT_images), axis=0)
        labels = np.concatenate((hupu_labels, SCUT_labels), axis=0)
        return images, labels

    def _build_graph(self):
        resnet = ResNet50(include_top=False, pooling='avg', input_shape=(self.args.input_size, self.args.input_size, 3), weights='imagenet')
        self.model = Sequential()
        self.model.add(resnet)
        self.model.add(Dense(1))
        if self.mode == 'fine_tune':
            self.model.layers[0].trainable = False
            print(self.model.summary())
        if self.mode == 'train':
            # for i in self.model.layers[:-1 * self.args.fix_layers]:
            #     i.trainable = False
            print(self.model.summary())
            self.model.load_weights('fine_tune_best.h5')
        if self.mode == 'predict':
            self.model.load_weights('train_best.h5')
        self.model.compile(loss='mae', optimizer='Adam', metrics=[r_square])


    def fine_tune(self):
        self.mode = 'fine_tune'
        self._build_graph()
        x, y = self._build_train_data()
        model_ckpt = ModelCheckpoint(filepath='fine_tune_best.h5', save_best_only=True, mode='min', monitor='val_loss', verbose=1)
        early_stop = EarlyStopping( monitor='val_loss', patience=self.args.patience, verbose=1, mode='min')
        self.model.fit(x=x,
                    y=y,
                    batch_size=self.args.batch_size,
                    epochs=self.args.max_epochs,
                    verbose=1,
                    callbacks=[model_ckpt, early_stop],
                    validation_split=0.2,
                    shuffle=True)


    def train(self):
        self.mode = 'train'
        self._build_graph()
        x, y = self._build_train_data()
        model_ckpt = ModelCheckpoint(filepath='train_best.h5', save_best_only=True, mode='min', monitor='val_loss', verbose=1)
        early_stop = EarlyStopping( monitor='val_loss', patience=self.args.patience, verbose=1, mode='min')
        self.model.fit(x=x,
                    y=y,
                    batch_size=self.args.batch_size,
                    epochs=self.args.max_epochs,
                    verbose=1,
                    callbacks=[model_ckpt, early_stop],
                    validation_split=0.2,
                    shuffle=True)

    def predict(self, fname):
        import face_recognition
        self.mode = 'predict'
        self._build_graph()
        images = []

        image = face_recognition.load_image_file(fname)
        try:
            face_locations = face_recognition.face_locations(image)
            top, right, bottom, left = face_locations[0]
            face_image = image[top:bottom, left:right]
            image = Image.fromarray(face_image)
            images.append(np.array(image.convert('RGB').resize((self.args.input_size, self.args.input_size), Image.LANCZOS)) / 255.0)
        except Exception as e:
            pass
            
        images = np.array(images)
        print(self.model.predict(images))

if __name__ == '__main__':
    tf.device('/gpu:0')
    model = model(args)
    if args.mode == 'train':
        model.train()
    if args.mode == 'fine_tune':
        model.fine_tune()
    if args.mode == 'predict':
        model.predict(args.image)
    