import tensorflow as tf
import os
import numpy as np
from PIL import Image
import config
from utils import extract_number

class MyTFDataset(tf.keras.utils.Sequence):
    def __init__(self, root_A, root_B, batch_size, root_C='', root_D='', face_counter=10):
        self.root_A = root_A
        self.root_B = root_B
        self.root_C = root_C
        self.root_D = root_D
        print(root_A)
        print(root_B)
        print(len(os.listdir(root_A)))
        print(len(os.listdir(root_B)))
        self.A_images = sorted(os.listdir(root_A), key=lambda x: extract_number(x) if extract_number(x) is not None else float('inf'))
        self.B_images = sorted(os.listdir(root_B), key=lambda x: extract_number(x) if extract_number(x) is not None else float('inf'))
        print(len(self.A_images))
        print(len(self.B_images))
        print(len(self.B_images)//self.batch_size)
        # if root_C != '':   self.C_images = sorted(os.listdir(root_C), key=lambda x: extract_number(x) if extract_number(x) is not None else float('inf'))
        # if root_D != '':  self.D_images = sorted(os.listdir(root_D), key=lambda x: extract_number(x) if extract_number(x) is not None else float('inf'))
        self.batch_size = batch_size
        #TODO: 10 batch scenery => 1 batch face photo parameters NOT SET YET SINCE THERE IS NO DATA.  GOTTA BE ADJUSTED
        self.face_counter = face_counter
        self.face_index = 0

    def __len__(self):
        # return int((max(len(self.A_images), len(self.B_images)) // self.batch_size) * 1.1) # length for applying face data
        return max(len(self.A_images), len(self.B_images)) // self.batch_size

    def __getitem__(self, index):
        # TODO: FACE BATCH ALGO
        # if self.face_counter == 10:
            # normal_batch = self.C_images[self.face_index * self.batch_size: (self.face_index + 1) * self.batch_size]
            # cartoon_batch = self.D_images[self.face_index * self.batch_size: (self.face_index + 1) * self.batch_size]
            # normal_path = self.root_A
            # cartoon_path = self.root_B
            # self.face_counter = 0
            # self.face_index += 1
        # else:
        normal_batch = self.A_images[index * self.batch_size: (index + 1) * self.batch_size]
        cartoon_batch = self.B_images[index * self.batch_size: (index + 1) * self.batch_size]

        normal_path = self.root_A
        cartoon_path = self.root_B
        normal_data = []
        cartoon_data = []


        for normal_photo, cartoon_photo in zip(normal_batch, cartoon_batch):
            A_path = os.path.join(normal_path, normal_photo)
            B_path = os.path.join(cartoon_path, cartoon_photo)

            normal_photo = np.array(Image.open(A_path).convert("RGB"))
            cartoon_photo = np.array(Image.open(B_path).convert("RGB"))

            normal_photo = config.transform_train(normal_photo)
            cartoon_photo = config.transform_train(cartoon_photo)

            normal_data.append(normal_photo)
            cartoon_data.append(cartoon_photo)

        return np.array(normal_data), np.array(cartoon_data)


class MyTestTFDataset(tf.keras.utils.Sequence):
    def __init__(self, root_A, batch_size):
        self.root_A = root_A
        self.A_images = os.listdir(root_A)
        self.batch_size = batch_size

    def __len__(self):
        return len(self.A_images) // self.batch_size

    def __getitem__(self, index):
        A_batch = self.A_images[index * self.batch_size: (index + 1) * self.batch_size]
        A_data = []

        for A_img in A_batch:
            A_path = os.path.join(self.root_A, A_img)
            A_img = np.array(Image.open(A_path).convert("RGB"))
            A_img = config.transform_test(image=A_img)["image"]
            A_data.append(A_img)

        return np.array(A_data), A_batch