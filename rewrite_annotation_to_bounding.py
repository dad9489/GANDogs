from PIL import Image
import os
import copy
from google_api import find_dog

ignored = ['Images/n02086646-Blenheim_spaniel/n02086646_1077.jpg']


def get_bounding_dim(path):
    path = 'Annotation/'+path[7:-4]
    bounds = []
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            if 17 < i < 22:
                bounds.append(int(line[9:line.find('</')]))
    return bounds


def get_square_dim(path):
    img = Image.open(path)
    orig_width, orig_height = img.size
    orig_dim = orig_width, orig_height
    bounding_coord = get_bounding_dim(path)
    img.crop(bounding_coord).show()
    bounding_dim = (bounding_coord[2] - bounding_coord[0], bounding_coord[3] - bounding_coord[1])
    middle = (bounding_coord[0]+(bounding_dim[0]/2), bounding_coord[1]+(bounding_dim[1]/2))
    bigger = 0 if bounding_dim[0] > bounding_dim[1] else 1
    shorter_side_len = orig_dim[bigger-1]
    if shorter_side_len >= bounding_dim[bigger]:  # if there is enough space on the shorter side
        if bigger == 1:  # if x side is shorter
            bounding_coord[0] = int(middle[0] - (bounding_dim[1] / 2))
            bounding_coord[2] = int(middle[0] + (bounding_dim[1] / 2))
            if bounding_coord[2] > orig_dim[0]:  # we went off right of screen
                bounding_coord[0] -= bounding_coord[2] - orig_dim[0]
                bounding_coord[2] = orig_dim[0]
            elif bounding_coord[0] < 0:  # we went off left of screen
                bounding_coord[2] += -bounding_coord[0]
                bounding_coord[0] = 0
        else:  # if y side is shorter
            bounding_coord[1] = int(middle[1] - (bounding_dim[0] / 2))
            bounding_coord[3] = int(middle[1] + (bounding_dim[0] / 2))
            if bounding_coord[3] > orig_dim[1]:  # we went off bottom of screen
                bounding_coord[1] -= bounding_coord[3] - orig_dim[1]
                bounding_coord[3] = orig_dim[1]
            elif bounding_coord[1] < 0:  # we went off top of screen
                bounding_coord[3] += -bounding_coord[1]
                bounding_coord[1] = 0
    else:  # not enough space on shorter side, so include as much as we can and crop longer side
        # top_mid = int(shorter_side_len*0.25)
        if bigger == 1:  # if x side is shorter (prob dog standing)
            bounding_coord[3] = bounding_dim[0]+bounding_coord[1]
        else:  # if y side is shorter (prob dog laying down)
            bounding_coord_cpy = copy.deepcopy(bounding_coord)
            bounding_coord_cpy[2] = bounding_dim[1] + bounding_coord[0]  # crop to left side
            temp_filepath = 'Images/temp/__temp.jpg'
            img.crop(bounding_coord_cpy).convert('RGB').save(temp_filepath)
            if find_dog(temp_filepath):
                bounding_coord = bounding_coord_cpy
            else:
                bounding_coord[0] = bounding_coord[2] - bounding_dim[1]  # crop to right side
            os.remove(temp_filepath)
    return bounding_coord[0], bounding_coord[1], bounding_coord[2], bounding_coord[3]


if __name__ == '__main__':
    # src_directory = 'Images/'
    # save_directory = 'Images/cropped/'
    # print(get_min_dim(src_directory))
    path = 'Images/n02086079-Pekinese/n02086079_10721.jpg'  # TODO test this
    img = Image.open(path)
    img.show()
    resize_dim = get_square_dim(path)
    cropped = img.crop(resize_dim)
    cropped.show()
