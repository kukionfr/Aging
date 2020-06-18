

import tensorflow as tf
from tensorflow.keras.layers import Dense
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
import shutil
from time import sleep
from copy import deepcopy

#%%

#prevent pre-allocation of gpu memory.
#cudnn failed to initialize without it
#I hate tensorflow gpu support team
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

#%%

# check tf is imported correctly
AUTOTUNE = tf.data.experimental.AUTOTUNE
print("TensorFlow Version: ", tf.__version__)
print("Number of GPU available: ", len(tf.config.experimental.list_physical_devices("GPU")))


# mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])
mirrored_strategy = tf.distribute.MirroredStrategy()

#%%

IMG_HEIGHT = 100
IMG_WIDTH = 100
BATCH_SIZE = 32
val_fraction = 30
max_epochs=300
class_target_sizes = [300, 400] #old, young

#%%

def read_and_label(file_path):
    label = get_label(file_path)
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])
    img = occlude(img, file_path)
    return img, label

def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    return tf.reshape(tf.where(parts[-4] == CLASS_NAMES), [])

def occlude(image, file_path):
    maskpth = tf.strings.regex_replace(file_path, 'image', 'label')
    mask = tf.io.read_file(maskpth)
    mask = tf.image.decode_jpeg(mask, channels=1)
    mask = tf.image.convert_image_dtype(mask, tf.float16)
    mask = tf.image.resize(mask, [IMG_WIDTH, IMG_HEIGHT])
    mask = tf.math.greater(mask, 0.25)
    # invert mask
    # mask = tf.math.logical_not(mask)
    maskedimg = tf.where(mask, image, tf.ones(tf.shape(image)))
    return maskedimg

def balance(dataset_dir, class_target_sizes):
    balanced_ds = [0]
    buffer = np.max(class_target_sizes)*2
    for CLASS, n in zip(CLASS_NAMES, class_target_sizes):
        sections = [_ for _ in dataset_dir.glob(CLASS+'/*')]
        for idx, section in enumerate(sections):
            section = os.path.join(section, 'image/*.jpg')
            list_ds = tf.data.Dataset.list_files(section, shuffle=False)
            # downsample if too big,
            list_ds = (list_ds
                       .shuffle(len(list(list_ds)))
                       .take(n)
                       )
            labeled_ds = list_ds.map(read_and_label, num_parallel_calls=AUTOTUNE)
            labeled_ds_org = list_ds.map(read_and_label, num_parallel_calls=AUTOTUNE)
            # upsample using augmentation if too small
            sampleN = len(list(labeled_ds_org))
            while sampleN < n:
                labeled_ds_aug = (labeled_ds_org
                                  .shuffle(sampleN)
                                  .take(n-sampleN)
                                  .map(augment,num_parallel_calls=AUTOTUNE)
                                  )
                # original + augmented image
                labeled_ds = labeled_ds.concatenate(labeled_ds_aug)
                sampleN = len(list(labeled_ds))
            labeled_ds.shuffle(buffer)
            print(CLASS, ' sample size balanced to ', sampleN)
            # append
            if balanced_ds[0] == 0:
                balanced_ds[idx] = labeled_ds
            else:
                labeled_ds = balanced_ds[0].concatenate(labeled_ds)
                balanced_ds[0] = labeled_ds
    return balanced_ds[0]

def augment(image, label):
    degree=0.05
    image = tf.image.random_hue(image, max_delta=degree, seed=5)
    image = tf.image.random_contrast(image, 1-degree, 1+degree, seed=5)  # tissue quality
    image = tf.image.random_saturation(image, 1-degree, 1+degree, seed=5)  # stain quality
    image = tf.image.random_brightness(image, max_delta=degree)  # tissue thickness, glass transparency (clean)
    image = tf.image.random_flip_left_right(image, seed=5)  # cell orientation
    image = tf.image.random_flip_up_down(image, seed=5)  # cell orientation
    # image = tf.image.rot90(image, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))  # cell orientation
    # To rotate 100x100 image, you need to rotate 142x142 image and crop center 100x100
    return image, label

#%%

# list location of all training images
train_data_dir = '/home/kuki2070s2/Desktop/Synology/aging/data/cnn_dataset/train'
train_data_dir = pathlib.Path(train_data_dir)
CLASS_NAMES = np.array([item.name for item in train_data_dir.glob('*') if item.name != ".DS_store"])
CLASS_NAMES = sorted(CLASS_NAMES, key=str.lower) #sort alphabetically case-insensitive


#%%

train_labeled_ds = balance(train_data_dir, class_target_sizes)

#%%

# list_ds = tf.data.Dataset.list_files(str(train_data_dir/ '*/*/image/*'))
# train_labeled_ds = list_ds.map(read_and_label, num_parallel_calls=AUTOTUNE)

#%%

train_image_count = len(list(train_labeled_ds))
print('training set size : ', train_image_count)
val_image_count = train_image_count // 100 * val_fraction
print('validation size: ', val_image_count)
train_image_count2 = train_image_count-val_image_count
print('training set size after split : ', train_image_count2)
STEPS_PER_EPOCH = train_image_count2 // BATCH_SIZE
VALIDATION_STEPS = val_image_count // BATCH_SIZE
print('train step #',STEPS_PER_EPOCH)
print('validation step #',VALIDATION_STEPS)

#%%

plt.figure(figsize=(10,10))
for idx, elem in enumerate(train_labeled_ds.take(100)):
    img = elem[0]
    label = elem[1]
    ax = plt.subplot(10,10,idx+1)
    plt.imshow(img)
    plt.title(CLASS_NAMES[label].title())
    plt.axis('off')
plt.show()


train_ds = (train_labeled_ds
            .skip(val_image_count)
            # .shuffle(buffer_size=len(list(train_labeled_ds)))
            # .shuffle(buffer_size=1000)
            .repeat()
            .batch(BATCH_SIZE)
            .prefetch(buffer_size=AUTOTUNE)
            )

#%%

val_ds = (train_labeled_ds
          .take(val_image_count)
          .repeat()
          .batch(BATCH_SIZE)
          .prefetch(buffer_size=AUTOTUNE))

#%%

# below make run out of ram
print('training set size : ', len(list(train_labeled_ds))-val_image_count)
print('validation set size : ', val_image_count)

test_data_dir = '/home/kuki2070s2/Desktop/Synology/aging/data/cnn_dataset/test'
test_data_dir = pathlib.Path(test_data_dir)

#%%

test_labeled_ds = balance(test_data_dir, class_target_sizes)

#%%

# test_list_ds = tf.data.Dataset.list_files(str(test_data_dir / '*/*/image/*'))
# test_labeled_ds = test_list_ds.map(read_and_label, num_parallel_calls=AUTOTUNE)

#%%

plt.figure(figsize=(10,10))
for idx,elem in enumerate(test_labeled_ds.take(100)):
    img = elem[0]
    label = elem[1]
    ax = plt.subplot(10,10,idx+1)
    plt.imshow(img)
    plt.title(CLASS_NAMES[label].title())
    plt.axis('off')
plt.show()

#%%

test_ds = (test_labeled_ds
           # .shuffle(buffer_size=len(list(test_labeled_ds)))
           # .shuffle(buffer_size=1000)
           .repeat()
           .batch(BATCH_SIZE)
           .prefetch(buffer_size=AUTOTUNE)  # time it takes to produce next element
           )

#%%

test_image_count = len(list(test_labeled_ds))
print('test set size : ', test_image_count)
TEST_STEPS = test_image_count // BATCH_SIZE
print('test step # ', TEST_STEPS)

#%%

# checkpoint_dir = "training_1"
# shutil.rmtree(checkpoint_dir, ignore_errors=True)

def get_callbacks(name):
    return [
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                         patience=50, restore_best_weights=True),
        # tf.keras.callbacks.TensorBoard(log_dir/name, histogram_freq=1),
        # tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir + "/{}/cp.ckpt".format(name),
        #                                    verbose=1,
        #                                    monitor='val_sparse_categorical_crossentropy',
        #                                    save_weights_only=True,
        #                                    save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy',
                                             factor=0.1, patience=20, verbose=1, mode='auto',
                                             min_delta=0.0001, cooldown=0, min_lr=0),
    ]

def compilefit(model, name, max_epochs, train_ds, val_ds):
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model_history = model.fit(train_ds,
                              steps_per_epoch=STEPS_PER_EPOCH,
                              epochs=max_epochs,
                              verbose=1,
                              validation_data=val_ds,
                              callbacks=get_callbacks(name),
                              validation_steps=VALIDATION_STEPS,
                              use_multiprocessing=True
                              )
    namename = os.path.dirname(name)
    if not os.path.isdir(os.path.abspath(namename)):
        os.mkdir(os.path.abspath(namename))
    if not os.path.isdir(os.path.abspath(name)):
        os.mkdir(os.path.abspath(name))
    if not os.path.isfile(pathlib.Path(name) / 'full_model.h5'):
        try:
            model.save(pathlib.Path(name) / 'full_model.h5')
        except:
            print('model not saved?')

    return model_history

def plotdf(dfobj, condition, repeat, lr=None):
    # pd.DataFrame(dfobj).plot(title=condition+repeat)
    dfobj1 = dfobj.copy()
    dfobj.pop('lr')
    dfobj.pop('loss')
    dfobj.pop('val_loss')
    pd.DataFrame(dfobj).plot(title=condition+'_'+repeat)
    plt.savefig('cnn/'+condition+'/'+repeat+'t1_accuracy.png')
    dfobj1.pop('lr')
    dfobj1.pop('accuracy')
    dfobj1.pop('val_accuracy')
    pd.DataFrame(dfobj1).plot(title=condition+'_'+repeat)
    plt.savefig('cnn/'+condition+'/'+repeat+'t1_loss.png')
    plt.show()

def evaluateit(network,networkname,repeat, train_ds, val_ds, test_ds):
    histories[networkname] = compilefit(network, 'cnn/'+networkname+'/'+repeat, max_epochs, train_ds, val_ds)
    results = network.evaluate(test_ds, steps=TEST_STEPS)
    plotdf(histories[networkname].history,networkname,repeat)
    print('test acc', results[-1] * 100)

def load_dataset(dataset_dir):
    dataset_dir = pathlib.Path(dataset_dir)
    test_image_count2 = len(list(dataset_dir.glob('image/*.jpg')))
    list_ds = tf.data.Dataset.list_files(str(dataset_dir / 'image/*.jpg'))
    labeled_ds = list_ds.map(read_and_label, num_parallel_calls=AUTOTUNE)
    return labeled_ds, test_image_count2

def evalmodels(path, model):
    datasett, datasettsize = load_dataset(path)
    print('datasetsize',datasettsize)
    results = model.evaluate(datasett.batch(1000))
    aa.append(np.around(results[-1] * 100, decimals=1))

#%%

histories = {}

#%%

def ds_resize(image,label):
    # image = tf.image.resize(image,[96,96])
    image = tf.image.central_crop(image,0.96)
    return image,label

#%%

train_ds_96 = train_ds.map(ds_resize, num_parallel_calls=AUTOTUNE)
val_ds_96 = val_ds.map(ds_resize, num_parallel_calls=AUTOTUNE)
test_ds_96 = test_ds.map(ds_resize, num_parallel_calls=AUTOTUNE)

#%%

# # Incompatible shapes: [64,1] vs. [64,3,3]
# for trial in ['t1','t2','t3','t4','t5']:
#     #min input size 75x75
#     DenseNet121_base = tf.keras.applications.DenseNet121(input_shape=(100, 100, 3),
#                                                 pooling=None,
#                                                 include_top=False,
#                                                 weights='imagenet'
#                                                 )
#     DenseNet121 = tf.keras.Sequential([
#         DenseNet121_base,
#         Dense(2, activation='softmax')
#     ])
#     evaluateit(DenseNet121,'DenseNet121',trial,train_ds,val_ds,test_ds)

#%%

trials = ['t4','t5']
trials = [_+'_cell' for _ in trials]

#
# #%%
#
# for trial in trials:
#     # shape 96, 128, 160, 192, 224
#     with mirrored_strategy.scope():
#         MobileNetV2_base = tf.keras.applications.MobileNetV2(input_shape=(96, 96, 3),
#                                                         pooling='avg',
#                                                         include_top=False,
#                                                         weights='imagenet'
#                                                         )
#         MobileNetV2 = tf.keras.Sequential([
#             MobileNetV2_base,
#                 Dense(2, activation='softmax')
#             ])
#         evaluateit(MobileNetV2,'MobileNetV2',trial,train_ds_96,val_ds_96,test_ds_96)
#
# #%%
#
# for trial in trials:
#     # min input 32x32
#     with mirrored_strategy.scope():
#         ResV2_base = tf.keras.applications.ResNet50V2(input_shape=(100, 100, 3),
#                                                     pooling='avg',
#                                                     include_top=False,
#                                                     weights='imagenet'
#                                                     )
#         ResV2 = tf.keras.Sequential([
#             ResV2_base,
#             Dense(2, activation='softmax')
#         ])
#         evaluateit(ResV2,'Res50V2',trial,train_ds,val_ds,test_ds)
# #%%
#
# for trial in trials:
#     # min input 32x32
#     with mirrored_strategy.scope():
#         Res101V2_base = tf.keras.applications.ResNet101V2(input_shape=(100, 100, 3),
#                                                     pooling='avg',
#                                                     include_top=False,
#                                                     weights='imagenet'
#                                                     )
#         Res101V2 = tf.keras.Sequential([
#             Res101V2_base,
#             Dense(2, activation='softmax')
#         ])
#         evaluateit(Res101V2,'Res101V2',trial,train_ds,val_ds,test_ds)
# #%%
#
# for trial in trials:
#     #min input size 75x75
#     with mirrored_strategy.scope():
#         IncV3_base = tf.keras.applications.InceptionV3(input_shape=(100, 100, 3),
#                                                     pooling=None,
#                                                     include_top=False,
#                                                     weights='imagenet'
#                                                     )
#         IncV3 = tf.keras.Sequential([
#             IncV3_base,
#             Dense(2, activation='softmax')
#         ])
#         evaluateit(IncV3,'IncV3',trial,train_ds,val_ds,test_ds)
#
# #%%

for trial in trials:
    with mirrored_strategy.scope():
        #min input size 75x75
        InceptionResNetV2_base = tf.keras.applications.InceptionResNetV2(input_shape=(100, 100, 3),
                                                    pooling=None,
                                                    include_top=False,
                                                    weights='imagenet'
                                                    )
        InceptionResNetV2 = tf.keras.Sequential([
            InceptionResNetV2_base,
            Dense(2, activation='softmax')
        ])
        evaluateit(InceptionResNetV2,'InceptionResNetV2',trial,train_ds,val_ds,test_ds)

#%%

# def load_compile(net):
#     model = tf.keras.models.load_model('cnn/' + net + '/full_model.h5', compile=False)
#     model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#                   optimizer=tf.keras.optimizers.Adam(),
#                   metrics=['accuracy'])
#     return model

#%%

# ResV2 = load_compile('Res50V2/t1')
# IncV3 = load_compile('IncV3/t1')
# InceptionResNetV2 = load_compile('InceptionResNetV2/t1')
# MobileNetV2 = load_compile('MobileNetV2/t1')

#%%

# ms = [ResV2, IncV3, InceptionResNetV2, MobileNetV2]
#
# for m in ms:
#     aa=[]
#     print('young train')
#     evalmodels('/home/kuki/Desktop/Synology/aging/data/cnn_dataset/train/young/sec001',m)
#     evalmodels('/home/kuki/Desktop/Synology/aging/data/cnn_dataset/train/young/sec003',m)
#     evalmodels('/home/kuki/Desktop/Synology/aging/data/cnn_dataset/train/young/sec007',m)
#     evalmodels('/home/kuki/Desktop/Synology/aging/data/cnn_dataset/train/young/sec010',m)
#     evalmodels('/home/kuki/Desktop/Synology/aging/data/cnn_dataset/train/young/sec016',m)
#     evalmodels('/home/kuki/Desktop/Synology/aging/data/cnn_dataset/train/young/sec019',m)
#     print('young test')
#     evalmodels('/home/kuki/Desktop/Synology/aging/data/cnn_dataset/test/young/sec023',m)
#     evalmodels('/home/kuki/Desktop/Synology/aging/data/cnn_dataset/test/young/sec025',m)
#     evalmodels('/home/kuki/Desktop/Synology/aging/data/cnn_dataset/test/young/sec029',m)
#     print('old train')
#     evalmodels('/home/kuki/Desktop/Synology/aging/data/cnn_dataset/train/old/sec031',m)
#     evalmodels('/home/kuki/Desktop/Synology/aging/data/cnn_dataset/train/old/sec037',m)
#     evalmodels('/home/kuki/Desktop/Synology/aging/data/cnn_dataset/train/old/sec041',m)
#     evalmodels('/home/kuki/Desktop/Synology/aging/data/cnn_dataset/train/old/sec045',m)
#     evalmodels('/home/kuki/Desktop/Synology/aging/data/cnn_dataset/train/old/sec049',m)
#     evalmodels('/home/kuki/Desktop/Synology/aging/data/cnn_dataset/train/old/sec062',m)
#     evalmodels('/home/kuki/Desktop/Synology/aging/data/cnn_dataset/train/old/sec068',m)
#     evalmodels('/home/kuki/Desktop/Synology/aging/data/cnn_dataset/train/old/sec070',m)
#     print('old test')
#     evalmodels('/home/kuki/Desktop/Synology/aging/data/cnn_dataset/test/old/sec076',m)
#     evalmodels('/home/kuki/Desktop/Synology/aging/data/cnn_dataset/test/old/sec078',m)
#     evalmodels('/home/kuki/Desktop/Synology/aging/data/cnn_dataset/test/old/sec082',m)
#     evalmodels('/home/kuki/Desktop/Synology/aging/data/cnn_dataset/test/old/sec088',m)
#     print(aa)

