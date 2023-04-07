import cv2
import os
from opt import parse_args
from sklearn.model_selection import train_test_split
from utils import read_data
args = parse_args()

# path to images
image_folder = os.path.join(args.data_path, 'images')

# read all image folders and sort
all_X_list = sorted(os.listdir(image_folder))

# train, valid split
train_image_list, test_image_list = \
    train_test_split(all_X_list, test_size=0.2, random_state = 0)
    
# train_image_list, valid_image_list = \dddddddd
#     train_test_split(train_image_list, test_size=0.7, random_state = 0)

# make 
train_dataset = read_data(train_image_list, args)
# test_dataset = read_data(test_image_list, args)

# train_dataset, valid_dataset = train_test_split(train_dataset, test_size = 0.3, stratify= train_dataset['Class0'])

for frame, X, Y, cls in train_dataset.values:
    if cls == 1:
        frame = os.sep.join(frame.split("/")[-2:])
        image = cv2.imread(args.data_path + '/images/' + frame)
        
        cv2.circle(image, (X,Y), 10, (0,0,255), -1)
        cv2.imshow(frame, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        