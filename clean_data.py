import os
from PIL import Image


def clean_folder(path):
    filepaths = os.listdir(path)
    do_not_work = []
    for file in filepaths:
        try:
            im = Image.open(os.path.join(path, file))
        except Exception as e:
            print(e)
            do_not_work.append(os.path.join(path, file))
    for path in do_not_work:
        os.remove(path)


if __name__ == "__main__":
    import sys
    datapath = sys.argv[1]
    print("[Start] Data Cleaning")
    for basepath in os.listdir(datapath): 
        for path in os.listdir(os.path.join(datapath, basepath)):
            # Remove corrupt images
            longpath = os.path.join(datapath, basepath, path)
            clean_folder(longpath)
            # Remove duplicate images
            os.system("image-cleaner %s" % longpath)
    print("[End] Data Cleaning")
