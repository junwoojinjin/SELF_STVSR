import os
import sys
import cv2
from tqdm import tqdm
import numpy as np
import glob
import os.path as osp
try:
    sys.path.append(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
    from data.util import imresize_np
except ImportError:
    pass


def generate_mod_LR_bic(up_scale, sourcedir, savedir, folder, sub_folder):
    # params: upscale factor, input directory, output directory

    saveHRpath = os.path.join(savedir, 'HR', 'x' + str(up_scale) , folder)
    saveLRpath = os.path.join(savedir, 'LR', 'x' + str(up_scale) , folder)
    saveBicpath = os.path.join(savedir, 'Bic', 'x' + str(up_scale), folder)

    saveHRpath_f = os.path.join(savedir, 'HR', 'x' + str(up_scale), folder)
    saveLRpath_f = os.path.join(savedir, 'LR', 'x' + str(up_scale), folder)
    saveBicpath_f = os.path.join(savedir, 'Bic', 'x' + str(up_scale), folder)

    """if not os.path.isdir(sourcedir):
        print(sourcedir)
        print('Error: No source data found')
        exit(0)"""
    if not os.path.isdir(savedir):
        os.mkdir(savedir)

    if not os.path.isdir(os.path.join(savedir, 'HR/x4')):
        os.mkdir(os.path.join(savedir, 'HR/x4'))
    if not os.path.isdir(os.path.join(savedir, 'LR/x4')):
        os.mkdir(os.path.join(savedir, 'LR/x4'))
    """if not os.path.isdir(os.path.join(savedir, 'Bic/x4')):
        os.mkdir(os.path.join(savedir, 'Bic/x4'))"""

    if not os.path.isdir(saveHRpath_f):
        os.mkdir(saveHRpath_f)
    if not os.path.isdir(saveLRpath_f):
        os.mkdir(saveLRpath_f)
    """if not os.path.isdir(saveBicpath_f):
        os.mkdir(saveBicpath_f)"""


    if not os.path.isdir(saveHRpath):
        os.mkdir(saveHRpath)
    else:
        print('It will cover ' + str(saveHRpath))

    if not os.path.isdir(saveLRpath):
        os.mkdir(saveLRpath)
    else:
        print('It will cover ' + str(saveLRpath))

    """if not os.path.isdir(saveBicpath):
        os.mkdir(saveBicpath)
    else:
        print('It will cover ' + str(saveBicpath))"""

    # read image
    image = cv2.imread(sourcedir)
    filename = sourcedir.split('/')[-1]
    width = int(np.floor(image.shape[1] / up_scale))
    height = int(np.floor(image.shape[0] / up_scale))
    # modcrop
    if len(image.shape) == 3:
        image_HR = image[0:up_scale * height, 0:up_scale * width, :]
    else:
        image_HR = image[0:up_scale * height, 0:up_scale * width]

    # LR
    image_LR = imresize_np(image_HR, 1 / up_scale, True)
    # bic
    # image_Bic = imresize_np(image_LR, up_scale, True)
    cv2.imwrite(os.path.join(saveHRpath, filename), image_HR)
    cv2.imwrite(os.path.join(saveLRpath, filename), image_LR)


""" filepaths = [f for f in os.listdir(sourcedir) if f.endswith('.png')]
    num_files = len(filepaths)

    # prepare data with augementation
    for i in range(num_files):
        filename = filepaths[i]
        # read image
        image = cv2.imread(os.path.join(sourcedir, filename))

        width = int(np.floor(image.shape[1] / up_scale))
        height = int(np.floor(image.shape[0] / up_scale))
        # modcrop
        if len(image.shape) == 3:
            image_HR = image[0:up_scale * height, 0:up_scale * width, :]
        else:
            image_HR = image[0:up_scale * height, 0:up_scale * width]

        # LR
        image_LR = imresize_np(image_HR, 1 / up_scale, True)
        # bic
        #image_Bic = imresize_np(image_LR, up_scale, True)

        cv2.imwrite(os.path.join(saveHRpath, filename), image_HR)
        cv2.imwrite(os.path.join(saveLRpath, filename), image_LR)
        #cv2.imwrite(os.path.join(saveBicpath, filename), image_Bic)
"""

if __name__ == "__main__":
    txt_file = '/mnt/ssd0/junwoojin/datas/Vid4/foliage.txt'
    img_folder = '/mnt/ssd0/junwoojin/datas/Vid4/'
    result_folder = '/mnt/ssd0/junwoojin/datas/Vid4_SR/'
    with open(txt_file) as f:
        train_l = f.readlines()
        train_l = [v.strip() for v in train_l]

    all_img_list = []
    print(train_l[0])
    for line in tqdm(train_l):

        folder = line.split('/')[0]
        sub_folder = line.split('/')[1]
        """if int(folder) < 16:
            continue"""
        generate_mod_LR_bic(4, osp.join(img_folder,folder,sub_folder), result_folder, folder, sub_folder)

