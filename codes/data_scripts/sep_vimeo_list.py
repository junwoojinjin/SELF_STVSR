import os
import shutil

if __name__ == "__main__":
    inPath = '/mnt/ssd0/junwoojin/datas/vimeo_septuplet/sequences/'
    outPath = '/mnt/ssd0/junwoojin/datas/vimeo_septuplet/sequences/test/'
    guide = '/mnt/ssd0/junwoojin/datas/vimeo_septuplet/sep_testlist.txt'

    f = open(guide, "r")
    lines = f.readlines()

    if not os.path.isdir(outPath):
        os.mkdir(outPath)

    for l in lines:
        line = l.replace('\n', '')
        this_folder = os.path.join(inPath, line)
        dest_folder = os.path.join(outPath, line)
        print(this_folder)
        shutil.copytree(this_folder, dest_folder)
    print('Done')
