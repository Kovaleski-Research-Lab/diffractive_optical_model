import numpy as np
import matplotlib.pyplot as plt
from IPython import embed
from PIL import Image, ImageDraw


def create_rectangle(row, col, width, height, angle):
    rect = np.array([(0, 0), (width, 0), (width, height), (0, height), (0, 0)])
    theta = (np.pi / 180) * angle
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    offset = np.array([col,row])
    transformed_rect = np.dot(rect, R) + offset
    return transformed_rect
    

def create_border(shape:tuple, width:int):
    dim_row, dim_col = shape
    left = create_rectangle(0, 0, width, dim_row, 0)
    top = create_rectangle(0, 0, dim_col, width, 0)
    right = create_rectangle(0, dim_col-width, width, dim_row,0)
    bot = create_rectangle(dim_row-width, 0, dim_col, width, 0)
    return [left, top, right, bot] 


def create_cross(row:int, col:int, len_row:int, len_col:int, width:int):
    #row and col define the bottom left of the square the cross is in
    vert = create_rectangle(row, col + (len_row - width//2),  len_col*2, width, 90)
    horz = create_rectangle(row - (len_col + width//2) , col, len_row*2, width, 0)
    return [vert, horz]


def create_flat(shape:tuple, value:int):
    return np.ones(shape) * value

if __name__ == "__main__":
    data = np.zeros((1080,1920), dtype='uint8')
    shape_data = data.shape

    test_grid = Image.fromarray(data)

    #rect = create_rectangle(240, 500, 100, 300, 0)
    #draw = ImageDraw.Draw(test_grid)
    #draw.polygon([tuple(p) for p in rect], fill=1)

    for r in create_border(data.shape, 30):
        draw = ImageDraw.Draw(test_grid)
        draw.polygon([tuple(p) for p in r], fill=1)

    for r in create_cross(1080,0,1920//2,1080//2,15):
        draw = ImageDraw.Draw(test_grid)
        draw.polygon([tuple(p) for p in r], fill=1)


    test_grid = np.asarray(test_grid) * 32
    print(np.max(test_grid))
    print(np.min(test_grid))
    test_grid = Image.fromarray(test_grid)
    print(np.max(test_grid))
    print(np.min(test_grid))

    test_grid.save('../slm_testPatterns/calibration_pattern.bmp')

    grid = np.asarray(test_grid)
