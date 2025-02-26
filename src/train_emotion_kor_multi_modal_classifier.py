# referenced from https://github.com/oarriaga/face_classification

import os
os.environ['CUDA_VISIBLE_DEVICES']='1,2,3,6,7'


from datetime import datetime

import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras as keras

#from models.cnn import mini_XCEPTION
from models.cnn import mini_XCEPTION_with_attention
from utils.datasets import DataManager
from utils.datasets import split_data
from utils.preprocessor import preprocess_input

mirrored_strategy = tf.distribute.MirroredStrategy()

# parameters
#NUM_GPU = 6
#batch_size = 32 * NUM_GPU
batch_size = 32
num_epochs = 1000
input_shape = (48, 48, 1)
validation_split = .2
verbose = 1
num_classes = 7
patience = 500

now = datetime.now()
current = now.strftime('%Y%m%d%H%M%S')

base_path = '../trained_models/kor_multi_modal_emotion_model_frontal_face'+current+'/'
print("train model path ", base_path)

# data generator
data_generator = ImageDataGenerator(
                        featurewise_center=False,
                        featurewise_std_normalization=False,
                        rotation_range=10,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        zoom_range=.1,
                        horizontal_flip=True)

with mirrored_strategy.scope():
        # model parameters/compilation
        model = mini_XCEPTION_with_attention(input_shape, num_classes)
        model.compile(optimizer='adam', loss='categorical_crossentropy',
                metrics=['accuracy'])
model.summary()

datasets = ['kor_multi_modal']
for dataset_name in datasets:
        print('Training dataset:', dataset_name)

        # callbacks
        os.makedirs(base_path, exist_ok=True)
        log_file_path = base_path + dataset_name + '_emotion_training.log'
        csv_logger = CSVLogger(log_file_path, append=False)
        early_stop = EarlyStopping('val_loss', patience=patience)
        reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1,
                                        patience=int(patience/4), verbose=1)
        trained_models_path = base_path + dataset_name + '_mini_XCEPTION'
        model_names = trained_models_path + '.{epoch:02d}-{val_accuracy:.2f}.hdf5'
        model_checkpoint = ModelCheckpoint(model_names, 'val_loss', verbose=1,
                                                        save_best_only=False)
        callbacks = [model_checkpoint, csv_logger, early_stop, reduce_lr]

        # loading dataset
        data_loader = DataManager(dataset_name, image_size=input_shape[:2])
        faces, emotions = data_loader.get_data()
        faces = preprocess_input(faces)
        num_samples, num_classes = emotions.shape
        train_data, val_data = split_data(faces, emotions, validation_split)
        train_faces, train_emotions = train_data

        # tf.data 설정
        ds = tf.data.Dataset.from_generator(lambda: data_generator.flow(train_faces, train_emotions, batch_size, shuffle=True),
                                        output_types=(tf.float32, tf.float32))
        ds = ds.cache()

        model.fit(ds, epochs=num_epochs, callbacks=callbacks, validation_data=val_data)
        # model.fit(ds,
        #         steps_per_epoch=len(train_faces) / batch_size,
        #         epochs=num_epochs, verbose=1, callbacks=callbacks,
        #         validation_data=val_data,
        #         use_multiprocessing=True,
        #         workers=6)

