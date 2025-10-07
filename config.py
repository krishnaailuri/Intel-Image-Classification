import os

BASE_DIR = '/Users/admin/Desktop/GIT/Riteesh/Image_classification/archive'
TRAIN_DIR = os.path.join(BASE_DIR, 'seg_train/seg_train')
TEST_DIR = os.path.join(BASE_DIR, 'seg_test/seg_test')
IMG_HEIGHT, IMG_WIDTH = 150, 150
BATCH_SIZE = 32
NUM_CLASSES = 6
EPOCHS = 30
