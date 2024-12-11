import sys
import os

current_file_dir = os.path.dirname(os.path.abspath(__file__))

# Define the relative path to the folder you want to add

common_folder = os.path.join(current_file_dir, "../common")
sys.path.append(common_folder)


from pl_modulo import print_model_info


if __name__ == '__main__':
    modelofilaname = sys.argv[1]

    print_model_info(modelofilaname)