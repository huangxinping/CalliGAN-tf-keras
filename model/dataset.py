import tensorflow as tf
import os


class CalliGANDataset(object):
    
    def __init__(self, batch_size=32, images_path=None, components_path=None, augment=False):
        assert images_path != None
        assert components_path != None
        
        self.images_path = images_path
        self.components_path = components_path
        
        self.n = len(os.listdir(self.images_path))
        print(f'{self.n} images found in {images_path}')
        
        self.hanzi2components = self.__get_component_config(components_path)
        
        self.ds = tf.data.Dataset.list_files(f'{images_path}/*.png')
        self.ds = self.ds.map(self.load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE).shuffle(self.n, reshuffle_each_iteration=True).batch(batch_size).repeat().prefetch(tf.data.AUTOTUNE)
    
    @property
    def dataset(self):
        return self.ds
    
    def __len__(self):
        return self.n
    
    def __get_component_config(self, components_path):
        hanzi2components = {}
        with open(components_path, encoding='utf-8') as f:
            for line in f.readlines():
                hanzi, components = line.strip().split('\t')
                hanzi2components[hanzi] = components.split(';')[0]
        return hanzi2components
        
    def convert_to_components(self, path):
        p = path.decode('utf-8')
        p = os.path.basename(p)
        hanzi = p.split('.')[0]
        hanzi = hanzi.split('_')[1]
        components = [int(value) for value in self.hanzi2components[hanzi].split(',')]
        components = tf.keras.preprocessing.sequence.pad_sequences([components], maxlen=28, padding='post')[0]
        return tf.cast(components, tf.float32)

    def convert_to_category(self, path):
        p = path.decode('utf-8')
        p = os.path.basename(p)
        font_index = int(p.split('_')[0])
        category = tf.keras.utils.to_categorical(font_index-1, num_classes=7)
        return category

    def augment(self, source, target, bold=False, rotate=False, blur=True):
        # TODO: may be has useful?
        if blur:
            source = tf.image.random_brightness(source, 0.4)
            target = tf.image.random_brightness(target, 0.4)
        source = tf.image.central_crop(source, central_fraction=0.9)
        return source, target

    def load_and_preprocess_image(self, path):
        # load image
        image = tf.io.read_file(path)
        image = tf.io.decode_png(image, channels=1)
        
        w = tf.shape(image)[1]
        w = w // 2
        source = image[:, :w, :]
        target = image[:, w:, :]
        
        source, target = tf.cast(source, tf.float32), tf.cast(target, tf.float32)
        # source, target = self.augment(source, target, bold=False, rotate=False, blur=True)
        
        source = (source/127.5)-1. 
        target = (target/127.5)-1.
        
        # get components
        components = tf.numpy_function(self.convert_to_components, [path], [tf.float32])[0]

        # one-hot encode category
        category = tf.numpy_function(self.convert_to_category, [path], [tf.float32])[0]
        
        return (components, source, category), target


if __name__ == '__main__':
    generator = CalliGANDataset(batch_size=32, images_path='datasets/images-3000', components_path='datasets/components/hanzi2components.txt')
    for item in generator.dataset.take(1):
        X, y = item
        co, s, c = X
        print(co.shape, s.shape, c.shape, y.shape)