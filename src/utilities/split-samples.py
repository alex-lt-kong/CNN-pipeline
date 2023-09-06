import os
import random
import shutil


def split_files(input_dir, output_dir_1, output_dir_2, split_ratio, seed=None) -> None:
        
    if os.path.exists(output_dir_1):
        shutil.rmtree(output_dir_1)
    os.makedirs(output_dir_1)
    if os.path.exists(output_dir_2):
        shutil.rmtree(output_dir_2)
    os.makedirs(output_dir_2)

    # Get a list of all files in the input directory
    files = os.listdir(input_dir)

    # Set the random seed
    random.seed(seed)

    # Calculate the number of files to copy to each output directory
    num_files = len(files)
    num_files_dir_1 = int(num_files * split_ratio)
    # num_files_dir_2 = num_files - num_files_dir_1

    # Randomly shuffle the list of files
    random.shuffle(files)

    # Copy files to the output directories based on the split ratio
    for i, file in enumerate(files):
        src = os.path.join(input_dir, file)
        if i < num_files_dir_1:
            dst = os.path.join(output_dir_1, file)
        else:
            dst = os.path.join(output_dir_2, file)
        shutil.copy2(src, dst)

    print(
        f'Splitting files from [{input_dir}] to '
        f'[{output_dir_1}]/[{output_dir_2}] completed successfully.'
    )


split_ratio = 0.9
random_seed = 123

input_directory = "/mnt/models/vgg16-based-pipeline/data/raw/0/"
output_directory_1 = "/mnt/models/vgg16-based-pipeline/data/training/0/"
output_directory_2 = "/mnt/models/vgg16-based-pipeline/data/validation/0/"
split_files(input_directory, output_directory_1, output_directory_2, split_ratio, random_seed)

input_directory = "/mnt/models/vgg16-based-pipeline/data/raw/1/"
output_directory_1 = "/mnt/models/vgg16-based-pipeline/data/training/1/"
output_directory_2 = "/mnt/models/vgg16-based-pipeline/data/validation/1/"
split_files(input_directory, output_directory_1, output_directory_2, split_ratio, random_seed)
