import os
import shutil
import glob

TRAIN_PATH = "./dataset/train/"
TEST_PATH = "./dataset/test/"



TRAIN_IMAGE_PATH = "./dataset/train_images/"
TRAIN_MASK_PATH = "./dataset/train_masks/"

TEST_IMAGE_PATH = "./dataset/test_images"
TEST_MASK_PATH = "./dataset/test_masks"


train_images = glob.glob(TRAIN_PATH + '/**/slice/*.tif', recursive=True)
train_masks = glob.glob(TRAIN_PATH + '/**/bone/*.tif', recursive=True)


test_images = glob.glob(TEST_PATH + '/**/slice/*.tif', recursive=True)
test_masks = glob.glob(TEST_PATH + '/**/bone/*.tif', recursive=True)


def copy(src, dest):
    for idx, file_name_src in enumerate(src):

        file_dir = os.path.basename((os.path.dirname(file_name_src)))  # Would be "Subfolder_with_patientID1"
        file_name = os.path.basename(file_name_src)  # Would be "2.dcm"

        file_name_dst = os.path.join(dest, f"{file_dir}_{idx}")  # Would be "/home/nponcian/Documents/folder_with_subfolders_dest/Subfolder_with_patientID1_2.dcm"

        shutil.copy2(file_name_src, file_name_dst)
        print(f"Copied:\n\tFr: {file_name_src}\n\tTo: {file_name_dst}")


src_list = [train_images, train_masks, test_images, test_masks]
dest_list = [TRAIN_IMAGE_PATH, TRAIN_MASK_PATH,TEST_IMAGE_PATH,TEST_MASK_PATH]

for i in range(4):
    copy(src_list[i], dest_list[i])