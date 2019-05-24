from PIL import Image
import os
import copy
from google_api import find_dog


# def get_min_dim(directory):
#     min_width = 100000000
#     min_height = 100000000
#     path = directory
#     for inner_folder in os.listdir(path):
#         path += inner_folder
#         for filename in os.listdir(path):
#             path += '/'+filename
#             img = Image.open(path)
#             width, height = img.size
#             # print('%s: (%d, %d)' % (filename, width, height))
#             if width < min_width: min_width = width
#             if height < min_height: min_height = height
#             path = path[0:len(directory+inner_folder)]
#         path = path[0:len(directory)]
#     return min_width, min_height


def crop_images_dumb(directory, min_dim, save_loc):
    path = directory
    for inner_folder in os.listdir(directory):
        path += inner_folder
        for filename in os.listdir(path):
            path += '/' + filename
            img = Image.open(path)
            width, height = img.size
            center = (width/2, height/2)
            dim = (min_dim[0], min_dim[1])
            crop_dim = (int(center[0]-(dim[0]/2)), int(center[1]-(dim[1]/2)), int(center[0]+(dim[0]/2)), int(center[1]+(dim[1]/2)))
            cropped = img.crop(crop_dim)
            cropped.convert('RGB').save(save_loc+filename)
            path = path[0:len(directory + inner_folder)]
        path = path[0:len(directory)]


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


def crop_images_smart(directory, save_loc):
    path = directory
    for inner_folder in os.listdir(directory):
        path += inner_folder
        for filename in os.listdir(path):
            path += '/' + filename
            img = Image.open(path)
            crop_coord = get_square_dim(path)
            cropped = img.crop(crop_coord)
            cropped.resize((150, 150)).convert('RGB').save(save_loc + filename)
            path = path[0:len(directory + inner_folder)]
        path = path[0:len(directory)]


def clean_images(directory):
    os.makedirs(directory + 'kept')
    os.makedirs(directory + 'removed')
    for filename in os.listdir(directory):
        if filename[-3:] == 'jpg':
            img = Image.open(directory + filename)
            if find_dog(directory+filename):
                img.save(directory + 'kept/' + filename)
            else:
                img.save(directory + 'removed/' + filename)


if __name__ == '__main__':
    src_directory = 'Images/'
    save_directory = 'Images/cropped/'
    # print(get_min_dim(src_directory))
    # min_dim = (97, 100)
    crop_images_smart(src_directory, save_directory)
    # clean_images(save_directory)
