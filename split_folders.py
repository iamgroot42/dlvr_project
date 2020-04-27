import os
import random
from shutil import copyfile
from tqdm import tqdm


if __name__ == "__main__":
    # Train-Test Split Ratio
    split_ratio = 0.7
    source_datapath = "./finegrained_data/"
    destn_datapath  = "./finegrained_data_split/"
    print("[Start] Data Split and Shifting")
    for basepath in os.listdir(source_datapath):
        bp = os.path.join(source_datapath, basepath)
        # Make directory for class
        os.mkdir(os.path.join(destn_datapath, basepath))
        # Make train, test directories inside class
        os.mkdir(os.path.join(destn_datapath, basepath, "train"))
        os.mkdir(os.path.join(destn_datapath, basepath, "test"))
        for folder in tqdm(os.listdir(bp)):
            concept = os.path.join(bp, folder)
            # Shuffle list
            files = os.listdir(concept)
            random.shuffle(files)
            # Split into train-test 
            split_point = int(len(files) * split_ratio)
            train_files, test_files = files[:split_point], files[split_point:]
            # Make directory for class
            dest_concept_path = os.path.join(destn_datapath, basepath)
            os.mkdir(os.path.join(dest_concept_path, "train", folder))
            os.mkdir(os.path.join(dest_concept_path, "test", folder))
            # Copy into destination folder to respectrive train/test directories
            for file in train_files:
                copyfile(os.path.join(concept, file), os.path.join(dest_concept_path, "train", folder, file))
            for file in test_files:
                copyfile(os.path.join(concept, file), os.path.join(dest_concept_path, "test", folder,file))
    print("[End] Data Split and Shifting")
