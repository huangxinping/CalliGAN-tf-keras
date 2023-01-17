import tensorflow as tf
import os
import numpy as np
import random
from PIL import Image, ImageFilter


class CalliGANGenerator(tf.keras.utils.Sequence):

    def __init__(self, batch_size=32, images_path=None, components_path=None, shuffle=True, augment=False, special_ids=None):
        super(CalliGANGenerator, self).__init__()
        self.shuffle = shuffle
        self.augment = augment
        self.batch_size = batch_size
        self.images_path = images_path
        if special_ids != None:
            self.image_paths = []
            special_ids = str(special_ids).split(',')
            for path in os.listdir(images_path):
                if (path.endswith('.jpg') or path.endswith('.png')) and ((path.split('_')[0] in special_ids)):
                    self.image_paths.append(path)
        else:
            self.image_paths = [path for path in os.listdir(images_path) if path.endswith('.jpg') or path.endswith('.png')]
        self.n = len(self.image_paths)
        self.components_path = components_path
        self.hanzi2components = self.__get_component_config()
        self.max = self.__len__()
        self.iter = 0
        self._debug_image_path = None
        self.on_epoch_end()
        print(f'{self.n} images found in {images_path}')

    def __get_component_config(self):
        hanzi2components = {}
        with open(self.components_path, encoding='utf-8') as f:
            for line in f.readlines():
                hanzi, components = line.strip().split('\t')
                hanzi2components[hanzi] = components.split(';')[0]
        return hanzi2components
    
    def __next__(self):
        if self.iter >= self.max:
            self.iter = 0
        result = self.__getitem__(self.iter)
        self.iter += 1
        return result
    
    def __len__(self):
        return len(self.image_paths) // self.batch_size

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        image_paths_per_step = [self.image_paths[k] for k in indexes]
        X, y = self.__data_generation(image_paths_per_step)
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.image_paths))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    def __augment(self, source, target, bold=False, rotate=False, blur=True):
        img_A = tf.keras.preprocessing.image.array_to_img(source)
        img_B = tf.keras.preprocessing.image.array_to_img(target)
        w, h = img_A.size
        if bold:
            multiplier = random.uniform(1.0, 1.2)
        else:
            multiplier = random.uniform(1.0, 1.05)
        # add an eps to prevent cropping issue
        nw = int(multiplier * w) + 1
        nh = int(multiplier * h) + 1

        # Used to use Image.BICUBIC, change to ANTIALIAS, get better image.
        img_A = img_A.resize((nw, nh), Image.Resampling.LANCZOS)
        img_B = img_B.resize((nw, nh), Image.Resampling.LANCZOS)

        shift_x = random.randint(0, max(nw - w - 1, 0))
        shift_y = random.randint(0, max(nh - h - 1, 0))

        img_A = img_A.crop((shift_x, shift_y, shift_x + w, shift_y + h))
        img_B = img_B.crop((shift_x, shift_y, shift_x + w, shift_y + h))

        if rotate and random.random() > 0.9:
            angle_list = [0, 180]
            random_angle = random.choice(angle_list)
            fill_color = 255
            img_A = img_A.rotate(random_angle, resample=Image.BILINEAR, fillcolor=fill_color)
            img_B = img_B.rotate(random_angle, resample=Image.BILINEAR, fillcolor=fill_color)

        if blur and random.random() > 0.8:
            sigma_list = [1, 1.5, 2]
            sigma = random.choice(sigma_list)
            img_A = img_A.filter(ImageFilter.GaussianBlur(radius=sigma))
            img_B = img_B.filter(ImageFilter.GaussianBlur(radius=sigma))
            
        return tf.keras.preprocessing.image.img_to_array(img_A), tf.keras.preprocessing.image.img_to_array(img_B)
    
    def load_all(self, path) -> tuple:
        # load image
        image = tf.keras.utils.load_img(os.path.join(self.images_path, path), color_mode="grayscale")
        image = tf.keras.preprocessing.image.img_to_array(image)
        
        source = image[:, :256, :]
        target = image[:, 256:, :]
#         target[target<np.mean(target)] = 0
        if self.augment:
            source, target = self.__augment(source, target)
        source = (source/127.5)-1. # Make image zero centered and in between (-1, 1), countpart tanh activation functoion
        target = (target/127.5)-1.
        
        # get components
        hanzi = path.split('.')[0]
        hanzi = hanzi.split('_')[1]
        components = [int(value) for value in self.hanzi2components[hanzi].split(',')]
        components = tf.keras.preprocessing.sequence.pad_sequences([components], maxlen=28, padding='post')[0]
        components = components.astype(np.float32)

        # one-hot encode category
        font_index = int(path.split('_')[0])
        category = tf.keras.utils.to_categorical(font_index-1, num_classes=7)
#         print(f'{hanzi} {components} {category}')
        
        return components, source, category, target

    def __data_generation(self, image_paths_per_step):
        components = []
        source_images = []
        categories = []
        target_images = []
        for image_path in image_paths_per_step:
            self._debug_image_path = image_path
            co, s, c, t = self.load_all(image_path)
            components.append(co)
            source_images.append(s)
            categories.append(c)
            target_images.append(t)
        components = np.asarray(components, dtype=np.float32)
        source_images = np.asarray(source_images, dtype=np.float32)
        categories = np.asarray(categories, dtype=np.float32)
        target_images = np.asarray(target_images, dtype=np.float32)
        return (components, source_images, categories), target_images
    
    
if __name__ == '__main__':
    gen = CalliGANGenerator(batch_size=32, images_path='datasets/images-3000', components_path='datasets/components/hanzi2components.txt', special_ids='1,2')
    X, y = next(gen)
    co, s, c = X
    print(co.shape, s.shape, c.shape, y.shape)