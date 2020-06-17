import tensorflow as tf
import os
import numpy as np
import pathlib
import pandas as pd

def read_and_label(file_path):
    label = get_label(file_path)
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [100, 100])
    # img = occlude(img, file_path)
    return img, label

def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    return tf.reshape(tf.where(parts[-4] == CLASS_NAMES), [])

def load_compile(net):
    model = tf.keras.models.load_model(os.path.join(*[model_dir,net,'full_model.h5']), compile=False)
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy'])
    return model

def evalmodels(path, model):
    datasett, datasettsize = load_dataset(path)
    print('folder : ',os.path.basename(path),' dataset size : ',datasettsize)
    results = model.evaluate(datasett.batch(64).prefetch(buffer_size=AUTOTUNE), verbose=0)
    aa.append(np.around(results[-1] * 100, decimals=1))


def load_dataset(dataset_dir):
    dataset_dir = pathlib.Path(dataset_dir)
    test_image_count2 = len(list(dataset_dir.glob('image/*.jpg')))
    list_ds = tf.data.Dataset.list_files(str(dataset_dir / 'image/*.jpg'))
    labeled_ds = list_ds.map(read_and_label, num_parallel_calls=AUTOTUNE)
    return labeled_ds, test_image_count2

# cudnn fail due to memory

# solution #1
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   try:
#     # Currently, memory growth needs to be the same across GPUs
#     for gpu in gpus:
#       tf.config.experimental.set_memory_growth(gpu, False)
#     logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#   except RuntimeError as e:
#     # Memory growth must be set before GPUs have been initialized
#     print(e)

# solution #2
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=9000)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)

AUTOTUNE = tf.data.experimental.AUTOTUNE
train_data_dir = '/home/kuki/Desktop/Synology/aging/data/cnn_dataset/train'
train_data_dir = pathlib.Path(train_data_dir)
CLASS_NAMES = np.array([item.name for item in train_data_dir.glob('*') if item.name != ".DS_store"])
testdir = '/home/kuki/Desktop/Synology/aging/data/cnn_dataset/test'

model_dir = 'cnn'
ms = ['Res50V2','IncV3', 'InceptionResNetV2']
ts = ['t'+str(_) for _ in range(1,16)]

csvname = 't1_t15.csv'
if os.path.exists(csvname):
    print('reading :', csvname)
    df = pd.read_csv(csvname,header=0,index_col=0)
else:
    print('empty')
    df = pd.DataFrame([],columns=[1,3,7,10,16,19,23,25,29,31,37,41,45,49,62,68,70,76,78,82,88])
for mm in ms:
    for t in ts:
        aa = []
        print(mm,t)
        m = load_compile(os.path.join(mm,t))
        print('young train')
        evalmodels(os.path.join(train_data_dir,'young/sec001'),m)
        evalmodels(os.path.join(train_data_dir,'young/sec003'),m)
        evalmodels(os.path.join(train_data_dir,'young/sec007'),m)
        evalmodels(os.path.join(train_data_dir,'young/sec010'),m)
        evalmodels(os.path.join(train_data_dir,'young/sec016'),m)
        evalmodels(os.path.join(train_data_dir,'young/sec019'),m)
        print('young test')
        evalmodels(os.path.join(testdir,'young/sec023'),m)
        evalmodels(os.path.join(testdir,'young/sec025'),m)
        evalmodels(os.path.join(testdir,'young/sec029'),m)
        print('old train')
        evalmodels(os.path.join(train_data_dir,'old/sec031'),m)
        evalmodels(os.path.join(train_data_dir,'old/sec037'),m)
        evalmodels(os.path.join(train_data_dir,'old/sec041'),m)
        evalmodels(os.path.join(train_data_dir,'old/sec045'),m)
        evalmodels(os.path.join(train_data_dir,'old/sec049'),m)
        evalmodels(os.path.join(train_data_dir,'old/sec062'),m)
        evalmodels(os.path.join(train_data_dir,'old/sec068'),m)
        evalmodels(os.path.join(train_data_dir,'old/sec070'),m)
        print('old test')
        evalmodels(os.path.join(testdir,'old/sec076'),m)
        evalmodels(os.path.join(testdir,'old/sec078'),m)
        evalmodels(os.path.join(testdir,'old/sec082'),m)
        evalmodels(os.path.join(testdir,'old/sec088'),m)
        df.loc[os.path.join(mm,t)]=aa
    df.to_csv(csvname)
    print('saved')
print(df)
