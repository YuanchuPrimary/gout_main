"""
args
"""

import argparse
import os


################### hyperparameters ###################

parser = argparse.ArgumentParser()

parser.add_argument('--id',
                    type=str, default='61_resnet_6')

parser.add_argument('--lr',
                    type=float, default=0.00001)

parser.add_argument('--epx',
                    type=int, default=50)

parser.add_argument('--batch_size',
                    type=int, default=64)

""" seed specifies the way the list is shuffled before split for cross validation, defines test data """
parser.add_argument('--seed',
                    type=int, default=41)

""" specifies the number of cross validation folds AND implicit the size of the validation data """
parser.add_argument('--cv_splits',
                    type=int, default=5)

""" train split """
parser.add_argument('--train_split',
                    type=int, default=0.9)

""" test split """
parser.add_argument('--test_split',
                    type=int, default=0.1)


parser.add_argument('--augm_hflip',
                    type=bool, default=False)
parser.add_argument('--augm_vflip',
                    type=bool, default=False)
parser.add_argument('--augm_crop',
                    type=bool, default=False)
parser.add_argument('--augm_rotate',
                    type=bool, default=False)

parser.add_argument('--data',
        type=str, default='gout_crop_700320') #### adjust path

""" train lat or all layers """
parser.add_argument('--feature_extract',
                    type=bool, default=False)

""" resnet18, resnet34, resnet50, resnet152, vgg11, densenet121, alexnet """
parser.add_argument('--model_name',
                    type=str, default='resnet_6')


parser.add_argument('--input_size',
                    type=int, default=600)


classes = list()
x = (os.path.join(os.getcwd(), parser.get_default('data')))
x= os.path.join(x,'train')
classes.append(0)
classes.append(1)
#classes.append(2)
#for a in os.listdir(x):
   # classes.append(a)
num_classes = len(classes)

args = parser.parse_args()
args = vars(args)


