import random
from .class_definitions import  COLORS
import numpy as np
from .drawing import draw_ellipse, draw_triangle, draw_circle, draw_square, draw_rectangle, IMSIZE
import matplotlib.pyplot as plt
import os
import shutil
import matplotlib

DISPATCHES = {
    'ellipse': draw_ellipse,
    'triangle': draw_triangle,
    'square': draw_square,
    'rectangle': draw_rectangle,
    'circle': draw_circle
}

SCALE = (IMSIZE / 64) ** 2


class GenerateShapes:
    AREA_MEANS =[SCALE * k for k in [1500, 1000, 600, 325, 125]]
    AREA_STDS = [SCALE * k for k in [50]*5]

    COLOR_MEANS = ['yellow', 'blue', 'green', 'purple', 'red']
    COLOR_STDS = [[25./255] * 3]*5

    # ORIENTATIONS = [100, 60, 45, 20, 0]
    # ORIENT_STDS = [5]*5

    SHAPE = ['circle', 'square', 'rectangle', 'ellipse', 'triangle']

    """
    Shapes can be characterized by: shape, color, size

    By default shape is the discriminative feature and color and size
    """
    def __init__(self, num_classes, use_angle):
        assert(1 <= num_classes <= 5)
        self.k = num_classes
        self.use_angle = use_angle

        # separately shuffle the three lists in order to make a different k be chosen every time
        random.shuffle(GenerateShapes.SHAPE)
        random.shuffle(GenerateShapes.COLOR_MEANS)
        random.shuffle(GenerateShapes.AREA_MEANS)


        c = list(zip(GenerateShapes.AREA_MEANS[:num_classes], GenerateShapes.COLOR_MEANS[:num_classes], GenerateShapes.SHAPE[:num_classes]))
        print(c)
        self.areas, self.colors, self.shapes = zip(*c)

        self.shape_to_idx = {shape: i for (i, shape) in enumerate(self.shapes)}


    def generate_canonical_instance(self, shape):
        idx = self.shape_to_idx[shape]

        area_mean = self.areas[idx]
        area_std = GenerateShapes.AREA_STDS[idx]
        area = np.clip(np.random.normal(area_mean, area_std), 50 * SCALE, 1600 * SCALE)

        color_mean = COLORS[self.colors[idx]]
        color_std = GenerateShapes.COLOR_STDS[idx]
        color = np.clip(np.random.normal(color_mean, color_std), 0, 1)

        if self.use_angle:
            angle = np.random.uniform(0, 180)
        else:
            angle = 0.

        img = DISPATCHES[shape](area, color, angle)

        return img

    def generate_noncanonical_instance(self, shape, independent):
        '''

        :param shape:
        :param independent: determines whether the noncanonical dimension should be chosen independently
        :return:
        '''
        idx = self.shape_to_idx[shape]

        area_idx = self.shape_to_idx[random.choice(self.shapes[:idx] + self.shapes[idx:])]

        if independent:
            color_idx = self.shape_to_idx[random.choice(self.shapes[:idx] + self.shapes[idx:])]
        else:
            color_idx = area_idx

        area_mean = self.areas[area_idx]
        area_std = GenerateShapes.AREA_STDS[area_idx]
        area = np.clip(np.random.normal(area_mean, area_std), 50 * SCALE, 1600 * SCALE)

        color_mean = COLORS[self.colors[color_idx]]
        color_std = GenerateShapes.COLOR_STDS[color_idx]
        color = np.clip(np.random.normal(color_mean, color_std), 0, 1)

        if self.use_angle:
            angle = np.random.uniform(0, 180)
        else:
            angle = 0

        img = DISPATCHES[shape](area, color, angle)

        return img

    def generate_samples(self, n_samples):
        samples = []
        per_category = int(n_samples/self.k)

        for shape in self.shapes:
            for i in range(per_category):
                samples.append(self.generate_canonical_instance(shape))
        return samples

    def generate_noncanonical_samples(self, n_samples, independent):
        samples = []
        per_category = int(n_samples / self.k)

        for shape in self.shapes:
            for i in range(per_category):
                samples.append(self.generate_noncanonical_instance(shape, independent))
        return samples


def create_dataset(path, name, categories, use_angle, num_canonical_samples, num_noncanonical_samples):
    name = '{}_{}_categories'.format(name, categories)
    if os.path.isdir('{}/{}'.format(path, name)):
        shutil.rmtree('{}/{}'.format(path, name))

    os.mkdir('{}/{}'.format(path, name))
    os.mkdir('{}/{}/{}'.format(path, name, 'canonical'))
    os.mkdir('{}/{}/{}'.format(path, name, 'noncanonical_independent'))
    os.mkdir('{}/{}/{}'.format(path, name, 'noncanonical_dependent'))

    generator = GenerateShapes(categories, use_angle=use_angle)

    d1 = generator.generate_samples(num_canonical_samples)
    plt.ioff()
    for idx, img_array in enumerate(d1):
        matplotlib.image.imsave(('{}/{}/{}/image_{}.png'.format(path, name, 'canonical', idx)), img_array)
    print('generated canonical samples')
    d2 = generator.generate_noncanonical_samples(num_noncanonical_samples, independent=True)
    for idx, img_array in enumerate(d2):
        matplotlib.image.imsave(('{}/{}/{}/image_{}.png'.format(path, name, 'noncanonical_independent', idx)), img_array)
    print('generated non-canonical independent samples')

    d3 = generator.generate_noncanonical_samples(num_noncanonical_samples, independent=False)
    for idx, img_array in enumerate(d3):
        matplotlib.image.imsave(('{}/{}/{}/image_{}.png'.format(path, name, 'noncanonical_dependent', idx)), img_array)
    print('generated canonical dependent samples')


if __name__ == '__main__':
    create_dataset('Shapes/Datasets', 'test_mk_data', 5, True, 10000, 5000)