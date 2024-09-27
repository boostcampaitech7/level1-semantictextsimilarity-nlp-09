import os
import shutil

from_dir = '/data/ephemeral/home/data'
to_dir = '/data/ephemeral/home/donghyuk/STS/data/raw'

if not os.path.exists(to_dir):
    os.makedirs(to_dir)


for file_name in os.listdir(from_dir):

    if file_name.endswith('.csv'):

        full_file_name = os.path.join(from_dir, file_name)    

        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, to_dir)
            print(f"Copied: {full_file_name} to {to_dir}")