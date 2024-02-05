#!/usr/bin/env python

import pickle
import sys

"""
    这个文件是用来查看*.pkl文件中的数据的维度的.

    UseCase: 
        `python show_degrees.py pr_text_embedding.pkl`
    Output:
        len(item) = 768
"""

def show_degrees(file: str):
    with open(file, 'rb') as f:
        print("loading file...")
        data = pickle.load(f)
        keys = list(data.keys())
        item = data[keys[0]]
        print(f'len(item) = {len(item)}')
        print(f'Size of data: {len(keys)}')


if __name__ == '__main__':
    show_degrees(sys.argv[1])