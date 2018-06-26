import os


inPath = '/home/zhanmusi/Documents/Data/Speech Commands Dataset/'
outPath = '/home/zhanmusi/Documents/Data/Speech Commands Dataset Downsampled/'


def loopFolders(path):
    folders = os.listdir(path)
    for folder in folders:
        loopFiles(folder)
    return


def loopFiles(path):
    return


def resampleFile(wav):
    return
