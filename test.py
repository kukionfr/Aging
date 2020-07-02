import tensorflow as tf
import tensorflow_hub as hub
import os
# os.environ["TFHUB_CACHE_DIR"] = '/home/kuki2070s2/PycharmProjects/Aging/hubcache'
print('downloading')
os.environ["TFHUB_CACHE_DIR"] = '/home/kuki2070s2/Desktop/TFHUB'
handle = hub.KerasLayer("https://tfhub.dev/google/imagenet/inception_v3/feature_vector/4",
                           trainable=True, arguments=dict(batch_norm_momentum=0.99)),  # Can be True, see below.
print('downloaded')
