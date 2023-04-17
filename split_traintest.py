import numpy as np
import os

import sys
import argparse


def get_args_parser():
    parser = argparse.ArgumentParser('split', add_help=False)
    parser.add_argument('--data_list_path', default='data/leaves_off/data_list.txt', type=str, help='path to data_list.txt')

    return parser

def main():
    #load data_list.txt
    f = open(args.data_list_path, 'r')

    #print out size of data_list.txt
    lines = f.readlines()
    print(f" size of dataset: {len(lines)}")


    #randomly split into train and test 9:1
    np.random.shuffle(lines)
    train_list = lines[:int(len(lines)*0.9)]
    test_list = lines[int(len(lines)*0.9):]

    print(f" size of train: {len(train_list)}")
    print(f" size of test: {len(test_list)}")


    #save train and test list to file
    with open('data/leaves_off/train_list.txt', 'w') as f:
        for item in train_list:
            f.write("%s" % item)

    with open('data/leaves_off/test_list.txt', 'w') as f:
        for item in test_list:
            f.write("%s" % item)

    print(f" -------- Done. Check for train_list.txt --------")
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser('split', parents=[get_args_parser()])
    args = parser.parse_args()
    main()
