import pandas as pd
import os
if __name__ == '__main__':
    file_names = ['train-00000-of-00007-342bf1c5df43ea16.parquet', 'train-00001-of-00007-ce8cae1ac51b900d.parquet', 'train-00002-of-00007-0c0432cda6ca5870.parquet'
    ,'train-00003-of-00007-6d3d7d54376e0dd5.parquet'
    ,'train-00004-of-00007-98f0165b64247bae.parquet'
    ,'train-00005-of-00007-c3ae3df137feae84.parquet'
    ,'train-00006-of-00007-6ad3fcf037ee2a98.parquet']


    if not os.path.exists('cartoon'):
        os.makedirs('cartoon')
    if not os.path.exists('normal'):
        os.makedirs('normal')

    df = pd.read_parquet(file_names[0], engine='pyarrow')
    for i, row in df.iterrows():

        with open(f'cartoon/cartoonized_image_{i}.jpg', 'wb') as f:
            f.write(row['cartoonized_image']['bytes'])

        with open(f'normal/original_image_{i}.jpg', 'wb') as f:
            f.write(row['original_image']['bytes'])

    df = pd.read_parquet(file_names[1], engine='pyarrow')
    for i, row in df.iterrows():
        i += 715

        with open(f'cartoon/cartoonized_image_{i}.jpg', 'wb') as f:
            f.write(row['cartoonized_image']['bytes'])

        with open(f'normal/original_image_{i}.jpg', 'wb') as f:
            f.write(row['original_image']['bytes'])

    df = pd.read_parquet(file_names[2], engine='pyarrow')
    for i, row in df.iterrows():
        i += 1430

        with open(f'cartoon/cartoonized_image_{i}.jpg', 'wb') as f:
            f.write(row['cartoonized_image']['bytes'])

        with open(f'normal/original_image_{i}.jpg', 'wb') as f:
            f.write(row['original_image']['bytes'])

    df = pd.read_parquet(file_names[3], engine='pyarrow')
    for i, row in df.iterrows():
        i += 2144

        with open(f'cartoon/cartoonized_image_{i}.jpg', 'wb') as f:
            f.write(row['cartoonized_image']['bytes'])

        with open(f'normal/original_image_{i}.jpg', 'wb') as f:
            f.write(row['original_image']['bytes'])

    df = pd.read_parquet(file_names[4], engine='pyarrow')
    for i, row in df.iterrows():
        i += 2858

        with open(f'cartoon/cartoonized_image_{i}.jpg', 'wb') as f:
            f.write(row['cartoonized_image']['bytes'])

        with open(f'normal/original_image_{i}.jpg', 'wb') as f:
            f.write(row['original_image']['bytes'])

    df = pd.read_parquet(file_names[5], engine='pyarrow')
    for i, row in df.iterrows():
        i += 3572

        with open(f'cartoon/cartoonized_image_{i}.jpg', 'wb') as f:
            f.write(row['cartoonized_image']['bytes'])

        with open(f'normal/original_image_{i}.jpg', 'wb') as f:
            f.write(row['original_image']['bytes'])

    df = pd.read_parquet(file_names[6], engine='pyarrow')
    for i, row in df.iterrows():
        i += 4286

        with open(f'cartoon/cartoonized_image_{i}.jpg', 'wb') as f:
            f.write(row['cartoonized_image']['bytes'])

        with open(f'normal/original_image_{i}.jpg', 'wb') as f:
            f.write(row['original_image']['bytes'])


