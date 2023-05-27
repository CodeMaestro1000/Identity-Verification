"""Script to download, unzip and prepare dataset dataset"""

import json, os, shutil, wget, zipfile, logging, cv2
from glob import glob
import numpy as np

logger = logging.getLogger('dev')
logger.setLevel(logging.INFO)

def read_image(img, label):
    '''
    Function to read in an image and a corresponding mask and then create the
    mask for each image
    '''
    image = cv2.imread(img)
    mask = np.zeros(image.shape, dtype=np.uint8)
    quad = json.load(open(label, 'r'))
    coords = np.array(quad['quad'], dtype=np.int32)
    cv2.fillPoly(mask, coords.reshape(-1, 4, 2), color=(255, 255, 255))
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = cv2.resize(mask, (mask.shape[1] // 2, mask.shape[0] // 2))
    image = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2))
    mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)[1]
    return image, mask

DOWNLOAD_PATH = 'downloads/'
TARGET_PATH = 'downloads/zip_data/'

if not os.path.exists(DOWNLOAD_PATH):
  os.mkdir(DOWNLOAD_PATH)
if not os.path.exists(TARGET_PATH):
  os.mkdir(TARGET_PATH)

logger.info('============== Now running: script to download and prepare data ========================')

download_links = [
    'ftp://smartengines.com/midv-500/dataset/12_deu_drvlic_new.zip',
    'ftp://smartengines.com/midv-500/dataset/13_deu_drvlic_old.zip',
    'ftp://smartengines.com/midv-500/dataset/14_deu_id_new.zip',
    'ftp://smartengines.com/midv-500/dataset/15_deu_id_old.zip',
    'ftp://smartengines.com/midv-500/dataset/16_deu_passport_new.zip',
    'ftp://smartengines.com/midv-500/dataset/17_deu_passport_old.zip'
]

# download files
for link in download_links:
  logger.info(f"Downloading: {link}")
  wget.download(link, TARGET_PATH)
  logger.info("Done...")
logger.info("======= All files downloaded ============")

# unzip all files
for fname in os.listdir(TARGET_PATH):
  with zipfile.ZipFile(TARGET_PATH + fname, 'r') as zip_ref:
    zip_ref.extractall(DOWNLOAD_PATH)

shutil.rmtree(TARGET_PATH) # Delete all zip files to save disk space

# Create new folder named 'dataset' to save all images
DATASET_PATH = 'dataset/'
IMAGE_PATH = 'dataset/IMG/images/' # needed to correctly feed data into train generator
MASK_PATH = 'dataset/MSK/masks/' # needed to correctly feed data into train generator

if not os.path.exists(DATASET_PATH): os.mkdir(DATASET_PATH)
if not os.path.exists(IMAGE_PATH): os.makedirs(IMAGE_PATH)
if not os.path.exists(MASK_PATH): os.makedirs(MASK_PATH)

FILE_SEARCH_REGEX = '/downloads/*/images/*/*'
MASK_SEARCH_REGEX = '/downloads/*/ground_truth/*/*'

filenames = sorted([name for name in glob(FILE_SEARCH_REGEX)])
masknames = sorted([name for name in glob(MASK_SEARCH_REGEX)])

# read all files
file_idx = 1
logger.info("Converting files to .png format....")
for img, label in zip(filenames, masknames):
  image, mask = read_image(img, label)
  fname = IMAGE_PATH + f'image_{file_idx}.png'
  maskname = MASK_PATH + f'mask_{file_idx}.png'
  cv2.imwrite(fname, image)
  cv2.imwrite(maskname, mask)
  if file_idx % 100 == 0:
    logger.info(f"====== {file_idx} files saved, last save: {maskname} ======")
  file_idx += 1
logger.info("Done.")