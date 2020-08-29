
import os
import numpy as np
from PIL import Image
import scipy
import pandas as pd
from matplotlib import pyplot as plt
import cv2
from time import time
import skimage
from skimage import io
from skimage.measure import label,find_contours,regionprops
from skimage.morphology import remove_small_objects
from joblib import Parallel, delayed, cpu_count
from copy import deepcopy
# custom function
from rgb2hed_v1 import separate_stains

# open svs and xml
tif_src = r'\\kukissd\research\lymphocyte\svs'
fns = os.listdir(tif_src)
fns = [os.path.splitext(fn)[0] for fn in fns]
for fn in fns:
    tif_path = os.path.join(*[tif_src,fn+'.tif'])
    dst = os.path.join(tif_src,fn)
    label_src =r'\\kukissd\research\lymphocyte\classified'
    label_path = os.path.join(*[label_src,fn+'.tif'])
    start = time()
    if os.path.exists(tif_path): svs = skimage.io.imread(tif_path)
    print("image loading: {:.2f} sec elapsed".format(time()-start))
    start = time()

    if os.path.exists(label_path): mask = skimage.io.imread(label_path)
    print("label loading: {:.2f} sec elapsed".format(time()-start))

    # find reticular dermis of each section
    start = time()
    mask[mask!=9]=0
    print("mask modification: {:.2f} sec elapsed".format(time()-start))


    region_mask = cv2.resize(mask,dsize=(svs.shape[1],svs.shape[0]))
    region_mask3d = np.repeat(region_mask[:, :, np.newaxis], 3, axis=2)

    region_mask[region_mask!=9]=0
    region_mask[region_mask==9]=1
    TA = np.sum(region_mask==1)

    svs[region_mask3d!=9] = 255
    region = svs/255
    region = region.astype(np.float32)

    # RGB to Haematoxylin-Eosin-DAB (HED) color space conversion.
    # Hematoxylin + Eosin + DAB

    start=time()
    # rgb_from_hed = np.array([[0.650, 0.704, 0.286],
    #                          [0.072, 0.990, 0.105],
    #                          [0.268, 0.570, 0.776]])
    rgb_from_hed = np.array([[0.650, 0.704, 0.286],
                             [0.072, 0.990, 0.105],
                             [0.0, 0.0, 0.0]])
    rgb_from_hed[2, :] = np.cross(rgb_from_hed[0, :], rgb_from_hed[1, :])
    hed_from_rgb = scipy.linalg.inv(rgb_from_hed)
    ihc_hed = separate_stains(region, hed_from_rgb)
    Hema = ihc_hed[:, :, 0]
    Hematoxylin = cv2.normalize(Hema, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    print("color deconvolution: {:.2f} sec elapsed".format(time()-start))
    #1 thresholding
    plt.figure(figsize=(12, 12))
    plt.hist(Hematoxylin.ravel(), 100, [1, 100])
    plt.xlim([1,100])
    plt.xticks(range(0,101,5))
    plt.show()
    threshold = input("Enter threshold (0-255):")
    threshold = int(threshold)
    Hematoxylin_temp = deepcopy(Hematoxylin)
    Hematoxylin_temp[Hematoxylin < threshold] = 0
    Hematoxylin_temp[Hematoxylin > threshold] = 1
    # label the objects
    labeled_bw = label(Hematoxylin_temp).astype(np.int64)
    #2 size filter
    start=time()
    remove_small_objects(labeled_bw,min_size=4,connectivity=1,in_place=True)
    print("number of nucleus as of now:",len(np.unique(labeled_bw)))
    print("size filter {:.2f} sec elapsed".format(time()-start))
    #2 check size filter and thresholding
    bw = (labeled_bw > 1) * 255
    plt.figure(figsize=(12,12))
    plt.imshow(bw[500:1500,500:1500],cmap='gray')
    plt.title(fn)
    plt.figure(figsize=(12, 12))
    plt.imshow(region[500:1500, 500:1500])
    plt.title(fn)
    plt.show()
    # 1 thresholding
    threshold = input("Enter threshold (0-255):")
    threshold = int(threshold)
    Hematoxylin[Hematoxylin < threshold] = 0
    Hematoxylin[Hematoxylin > threshold] = 1
    # label the objects
    labeled_bw = label(Hematoxylin).astype(np.int64)
    # 2 size filter
    start = time()
    remove_small_objects(labeled_bw, min_size=4, connectivity=1, in_place=True)
    print("number of nucleus as of now:", len(np.unique(labeled_bw)))
    print("size filter {:.2f} sec elapsed".format(time() - start))
    # 2 check size filter and thresholding
    bw = (labeled_bw > 1) * 255
    plt.figure(figsize=(12, 12))
    plt.imshow(bw[500:1500, 500:1500], cmap='gray')
    plt.title(fn)
    plt.figure(figsize=(12, 12))
    plt.imshow(region[500:1500, 500:1500])
    plt.title(fn)
    plt.show()
    #2 save bw before AR filter
    bw_img = Image.fromarray(bw).convert('1')
    bw_img.save(dst+ '_SZ_filtered.tif')
    #3 check AR
    plt.figure(figsize=(12, 12))
    props = regionprops(labeled_bw)
    ARs = [x['major_axis_length']/x['minor_axis_length'] if x['minor_axis_length']!=0 else 0 for x in props ]
    plt.hist(ARs, 40, [0, 4])
    plt.xlim([0,4])
    plt.show()
    #3 AR filter
    minAR = 0
    maxAR = input('max AR?:')
    maxAR = float(maxAR)
    start=time()
    def ARfilter(x):
        if (x['minor_axis_length']!=0): AR = x['major_axis_length']/x['minor_axis_length']
        else: AR = 0
        if AR<minAR: labeled_bw[labeled_bw==x.label]=0
        if AR>maxAR: labeled_bw[labeled_bw==x.label]=0
    Parallel(n_jobs=-4, prefer="threads")(delayed(ARfilter)(x) for x in props)
    print("number of nucleus as of now:", len(np.unique(labeled_bw)))
    print("AR filter {:.2f} sec elapsed".format(time()-start))
    #3 save bw before SF filter
    bw = (labeled_bw > 1) * 255
    bw_img = Image.fromarray(bw).convert('1')
    bw_img.save(dst+ '_AR_filtered.tif')
    #3 check AR
    plt.figure(figsize=(12, 12))
    props = regionprops(labeled_bw)
    SFs = [4*np.pi*x['area']/x['perimeter']**2 for x in props]
    plt.hist(SFs, 20, [0, 2])
    plt.xlim([0,2])
    plt.show()
    #4 SF filter (keep cells in the range)
    start = time()
    minSF = input("minSF?:")
    minSF = float(minSF)
    maxSF = 2
    for prop in props:
        SF = 4*np.pi*prop['area']/prop['perimeter']**2
        if SF<minSF: labeled_bw[labeled_bw==prop.label]=0;
        if SF>maxSF: labeled_bw[labeled_bw==prop.label]=0;
    print("number of nucleus as of now:", len(np.unique(labeled_bw)))
    print("SF filter {:.2f} sec elapsed".format(time() - start))
    #4 save bw before distance filter
    bw = (labeled_bw>0)*255
    bw_img = Image.fromarray(bw).convert('1')
    bw_img.save(dst+ '_SF_filtered.tif')
    #6 export datasheet
    prop = regionprops(labeled_bw)
    xs = [np.around(_['centroid'][0]) for _ in prop]
    ys = [np.around(_['centroid'][1]) for _ in prop]
    area = [np.sum(x._label_image[x._slice] == x.label) for x in prop]
    ARs = [np.around(_['major_axis_length']/_['minor_axis_length'],decimals=3) for _ in prop]
    SFs = [np.around(4*np.pi*_['area']/_['perimeter']**2,decimals=3) for _ in prop]
    dict = {'x':xs,'y':ys,'area':area,'aspect_ratio':ARs,'circularity':SFs}
    df = pd.DataFrame(dict)
    df.to_csv(dst+'_SF_filtered.csv')

    n = len(np.unique(labeled_bw))
    d = n / TA
    print(n,d)

