import os
import sys
sys.path.insert(0, '.')
from models.mini_resnet import ResNet as Net
import tensorflow as tf
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_image_from_df(df):
    pix_cols = [column for column in df.columns if 'pixel' in column]
    images = df[pix_cols].to_numpy()
    images = images.reshape(-1, 28, 28, 1)
    y = None
    if 'label' in df.columns:
        y = df['label'].to_numpy().reshape(-1)
        y = tf.convert_to_tensor(y)
    
    images = tf.convert_to_tensor(images)
    return images, y

def main():
    train = pd.read_csv("data/train.csv")
    train = train.sample(frac=1)
    test = pd.read_csv("data/test.csv")

    images_train, y_train = get_image_from_df(train)
    images_test, _ = get_image_from_df(test)

    train_datagen = ImageDataGenerator(rescale=1/255, validation_split=0.04)

    train_flow = train_datagen.flow(images_train,
                                    y_train,
                                    batch_size=200,
                                    subset='training')
    validation_flow = train_datagen.flow(images_train,
                                        y_train,
                                        batch_size=100,
                                        subset='validation')

    model = Net(10)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                loss='sparse_categorical_crossentropy',
                metrics=['sparse_categorical_accuracy'])

    model.build((None, 28, 28, 1))
    print(model.summary())

    history = model.fit(train_flow,
                        validation_data=validation_flow,
                        epochs=15)
    
    os.makedirs('checkpoints/')
    model.save('checkpoints/ckpt.h5')

if __name__=="__main__":
    main()