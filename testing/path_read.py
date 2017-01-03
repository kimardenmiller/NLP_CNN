import os

fname = '../saved_models/test.txt'
if os.path.isfile(fname):
    print("Found file '{}'".format(fname))
else:
    print('File not found')
    # with open(fname, 'r') as f:
    #     for line in f:
    #         print line