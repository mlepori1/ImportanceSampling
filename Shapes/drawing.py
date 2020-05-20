import numpy as np
from skimage.transform import rotate
from Shapes.class_definitions import SHAPES, COLORS
import matplotlib.pyplot as plt

IMSIZE = 64


def draw_shape(imsize, shape, color, angle, args):
    img = np.zeros((imsize, imsize, 3), dtype=np.double)

    rr, cc = SHAPES[shape](**args)
    rr = rr.astype(int)
    cc = cc.astype(int)
    img[rr, cc] = color

    img = rotate(img, angle)
    return img


def draw_ellipse(area, color, degrees=0):
    # ellipses will have constant major/minor ratio of 2:1
    major_axis = np.sqrt(2*area/np.pi)

    args = {
        'r': IMSIZE/2,
        'c': IMSIZE/2,
        'r_radius': major_axis/2,
        'c_radius': major_axis,
    }

    return draw_shape(IMSIZE, 'ellipse', color,  degrees, args)


def draw_circle(area, color, degrees=0):
    radius = np.sqrt(area/np.pi)
    args = {'radius': radius,
            'r': IMSIZE / 2,
            'c': IMSIZE / 2}
    return draw_shape(IMSIZE, 'circle', color, degrees, args)


def draw_rectangle(area, color, degrees=0):
    # squares will have 2:1 ratio
    length = np.sqrt(2*area)
    width = length/2
    CENTER = (IMSIZE/2, IMSIZE/2)
    LEFT_CORNER = (CENTER[0] - width/2, CENTER[1] - length/2)
    RIGHT_CORNER = (CENTER[0] + width/2, CENTER[1] + length/2)
    args = {
        'start': LEFT_CORNER,
        'end': RIGHT_CORNER,
    }
    return draw_shape(IMSIZE, 'rectangle', color, degrees, args)


def draw_square(area, color, degrees=0):
    # squares will have 2:1 ratio
    length = np.sqrt(area)
    width = length
    CENTER = (IMSIZE/2, IMSIZE/2)
    LEFT_CORNER = (CENTER[0] - width/2, CENTER[1] - length/2)
    RIGHT_CORNER = (CENTER[0] + width/2, CENTER[1] + length/2)
    args = {
        'start': LEFT_CORNER,
        'end': RIGHT_CORNER,
    }
    return draw_shape(IMSIZE, 'rectangle', color, degrees, args)


def draw_triangle(area, color, degrees=0):
    # equilateral triangles only
    length = np.sqrt(4 * area/np.sqrt(3))

    height = np.sqrt(3)/2 * length

    top = (IMSIZE/2 + (2/3) * height, IMSIZE/2)
    left = (IMSIZE/2 - (1/3) * height, IMSIZE/2 - length/2)
    right = (IMSIZE/2 - (1/3) * height, IMSIZE/2 + length/2)

    r = [top[0], left[0], right[0]]
    c = [top[1], left[1], right[1]]

    args = {
        'r': r,
        'c': c
    }

    return draw_shape(IMSIZE, 'triangle', color, degrees, args)


if __name__ == '__main__':
    # draw_ellipse(100, 'yellow', 0)
    # draw_circle(100, 'blue')
    # draw_rectangle(100, 'red', 0)
    # draw_square(100, 'purple', 0)
    area = 625 * np.sqrt(3) / 4
    fix, ax = plt.subplots(1, 1)
    img = draw_triangle(area, COLORS['green'], 0)
    ax.imshow(img)
    # ax.axis('off')
    plt.show()

