import random
from Shapes.class_definitions import COLORS
import numpy as np
from Shapes.drawing import draw_ellipse, draw_triangle, draw_circle, draw_square, draw_rectangle, IMSIZE
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


DEFAULT_AREA_MEANS = [SCALE * k for k in [1500, 1000, 600, 325, 125]]
DEFAULT_AREA_STDS = [SCALE * k for k in [50] * 5]

DEFAULT_COLOR_MEANS = ['yellow', 'blue', 'green', 'purple', 'red']
DEFAULT_COLOR_STDS = [[25. / 255] * 3] * 5

DEFAULT_SHAPES = ['circle', 'square', 'rectangle', 'ellipse', 'triangle']

class GenerateShapes:
    """
    Shapes can be characterized by: shape, color, size

    By default shape is the discriminative feature and color and size
    """
    def __init__(self, num_classes, use_angle=True, shapes=None, areas=None, area_stds=None, colors=None, color_stds=None, randomize=False):

        if shapes is None:
            self.SHAPE = DEFAULT_SHAPES
        else:
            self.SHAPE = shapes

        if areas is None:
            self.AREA_MEANS = DEFAULT_AREA_MEANS
        else:
            self.AREA_MEANS = areas

        if area_stds is None:
            self.AREA_STDS = DEFAULT_AREA_STDS
        else:
            self.AREA_STDS = area_stds

        if colors is None:
            self.COLOR_MEANS = DEFAULT_COLOR_MEANS
        else:
            self.COLOR_MEANS = colors

        if color_stds is None:
            self.COLOR_STDS = DEFAULT_COLOR_STDS
        else:
            self.COLOR_STDS = color_stds

        assert (1 <= num_classes <= len(self.SHAPE))
        self.k = num_classes
        self.use_angle = use_angle

        if randomize:
            # separately shuffle the three lists in order to make a different k be chosen every time
            random.shuffle(self.SHAPE)
            random.shuffle(self.COLOR_MEANS)
            random.shuffle(self.AREA_MEANS)


        c = list(zip(self.AREA_MEANS[:num_classes], self.COLOR_MEANS[:num_classes], self.SHAPE[:num_classes]))
        print(c)
        self.areas, self.colors, self.shapes = zip(*c)



    def generate_canonical_instance(self, idx):
        shape = self.SHAPE[idx]
        area_mean = self.areas[idx]
        area_std = self.AREA_STDS[idx]
        area = np.clip(np.random.normal(area_mean, area_std), 50 * SCALE, 1600 * SCALE)

        color_mean = COLORS[self.colors[idx]]
        color_std = self.COLOR_STDS[idx]
        color = np.clip(np.random.normal(color_mean, color_std), 0, 1)

        if self.use_angle:
            angle = np.random.uniform(0, 180)
        else:
            angle = 0.

        img = DISPATCHES[shape](area, color, angle)

        return img

    def generate_noncanonical_instance(self, idx, independent):
        '''

        :param shape:
        :param independent: determines whether the noncanonical dimension should be chosen independently
        :return:
        '''
        shape = self.SHAPE[idx]

        idxs = range(len(shape))
        area_idx = random.choice(idxs[:idx] + idxs[idx:])

        if independent:
            color_idx = random.choice(idxs[:idx] + idxs[idx:])
        else:
            color_idx = area_idx

        area_mean = self.areas[area_idx]
        area_std = self.AREA_STDS[area_idx]
        area = np.clip(np.random.normal(area_mean, area_std), 50 * SCALE, 1600 * SCALE)

        color_mean = COLORS[self.colors[color_idx]]
        color_std = self.COLOR_STDS[color_idx]
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

        for idx in range(len(self.shapes)):
            for i in range(per_category):
                samples.append(self.generate_canonical_instance(idx))
        return samples

    def generate_noncanonical_samples(self, n_samples, independent):
        samples = []
        per_category = int(n_samples / self.k)

        for idx in range(len(self.shapes)):
            for i in range(per_category):
                samples.append(self.generate_noncanonical_instance(idx, independent))
        return samples



def create_dataset(path, name, generator, categories, num_canonical_samples, num_noncanonical_samples=None):
    name = '{}_{}_categories'.format(name, categories)
    if os.path.isdir('{}/{}'.format(path, name)):
        shutil.rmtree('{}/{}'.format(path, name))

    os.mkdir('{}/{}'.format(path, name))
    os.mkdir('{}/{}/{}'.format(path, name, 'canonical'))



    d1 = generator.generate_samples(num_canonical_samples)
    plt.ioff()
    for idx, img_array in enumerate(d1):
        matplotlib.image.imsave(('{}/{}/{}/image_{}.png'.format(path, name, 'canonical', idx)), img_array)
    print('generated canonical samples')

    if num_noncanonical_samples is not None:
        os.mkdir('{}/{}/{}'.format(path, name, 'noncanonical_independent'))
        os.mkdir('{}/{}/{}'.format(path, name, 'noncanonical_dependent'))

        d2 = generator.generate_noncanonical_samples(num_noncanonical_samples, independent=True)

        for idx, img_array in enumerate(d2):
            matplotlib.image.imsave(('{}/{}/{}/image_{}.png'.format(path, name, 'noncanonical_independent', idx)), img_array)
        print('generated non-canonical independent samples')

        d3 = generator.generate_noncanonical_samples(num_noncanonical_samples, independent=False)
        for idx, img_array in enumerate(d3):
            matplotlib.image.imsave(('{}/{}/{}/image_{}.png'.format(path, name, 'noncanonical_dependent', idx)), img_array)
        print('generated canonical dependent samples')

if __name__ == '__main__':
    colors = ['blue', 'red', 'blue', 'red']
    color_stds = [0., 0., 0., 0.]

    shapes = ['square', 'square', 'triangle', 'triangle']

    areas = [1000, 1000, 1000, 1000]
    area_stds = [10, 10, 10, 10]

    categories = 4
    use_angle = True

    generator = GenerateShapes(categories, use_angle=use_angle, shapes=shapes, areas=areas, colors=colors, area_stds=area_stds, color_stds=color_stds, randomize=False)

    create_dataset('Datasets', 'test_mk_data', generator, categories, 10000)