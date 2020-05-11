import numpy as np
import cv2, re

""" Load depth 
"""
def load_depth_from_tiff(tiff_file_path, div_factor=1000.0):
    depth = cv2.imread(tiff_file_path, cv2.IMREAD_ANYDEPTH).astype(np.float32)
    return depth / div_factor


def load_depth_from_png(png_file_path, div_factor=1000.0):
    depth = cv2.imread(png_file_path, cv2.IMREAD_ANYDEPTH).astype(np.float32)
    return depth / div_factor


def load_depth_from_pgm(pgm_file_path, min_depth_threshold=None):
    """
    Load the depth map from PGM file
    :param pgm_file_path: pgm file path
    :return: depth map with 2D ND-array
    """
    raw_img = None
    with open(pgm_file_path, 'rb') as f:
        line = str(f.readline(), encoding='ascii')
        if line != 'P5\n':
            print('Error loading pgm, format error\n')

        line = str(f.readline(), encoding='ascii')
        max_depth = float(line.split(" ")[-1].strip())

        line = str(f.readline(), encoding='ascii')
        dims = line.split(" ")
        cols = int(dims[0].strip())
        rows = int(dims[1].strip())

        line = str(f.readline(), encoding='ascii')
        max_factor = float(line.strip())

        raw_img = np.frombuffer(f.read(cols * rows * np.dtype(np.uint16).itemsize), dtype=np.uint16).reshape((rows, cols)).astype(np.float32)
        raw_img *= (max_depth / max_factor)

    if min_depth_threshold is not None:
        raw_img[raw_img < min_depth_threshold] = min_depth_threshold
    return raw_img


def load_depth_from_pfm(ppm_file_path):
    """
    Load the file from PFM file
    :param ppm_file_path: pfm file path
    :return: depth map with 2D nd-array
    """
    with open(ppm_file_path, 'rb') as file:
        header = str(file.readline().decode('utf-8')).rstrip()

        if header == 'PF':
            color = True
        elif header == 'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')
        dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception('Malformed PFM header.')
        scale = float((file.readline().decode('utf-8')).rstrip())
        if scale < 0:  # little-endian
            data_type = '<f'
        else:
            data_type = '>f'  # big-endian
        data_string = file.read()
        data = np.fromstring(data_string, data_type)
        # data = np.fromfile(file, data_type)
        shape = (height, width, 3) if color else (height, width)
        data = np.reshape(data, shape)
        data = cv2.flip(data, 0)
        return data


""" Save Routines
"""
def save_depth_to_tiff(depth, tiff_file_path, div_factor=1000):
    depth = (depth*div_factor).astype(np.uint16)
    cv2.imwrite(tiff_file_path, depth)


def save_depth_to_pgm(depth, pgm_file_path):
    """
    Save the depth map to PGM file
    :param depth: depth map with 2D ND-array
    :param pgm_file_path: output file path
    """
    max_depth = np.max(depth)
    depth_copy = np.copy(depth)
    depth_copy = 65535.0 * (depth_copy / max_depth)
    depth_copy = depth_copy.astype(np.uint16)

    with open(pgm_file_path, 'wb') as f:
        f.write(bytes("P5\n", encoding="ascii"))
        f.write(bytes("# %f\n" % max_depth, encoding="ascii"))
        f.write(bytes("%d %d\n" % (depth.shape[1], depth.shape[0]), encoding="ascii"))
        f.write(bytes("65535\n", encoding="ascii"))
        f.write(depth_copy.tobytes())


def generate_inverse_depth_map(depth_map, min_truncate=0.0, max_truncate=1.0):
    """
    Reverse the depth map from d to 1/d
    :param depth_map: depth map with 2D ND-array
    :param min_truncate: min depth threshold value
    :param max_truncate:  max depth threshold value
    :return: inverse depth
    """
    depth_map_copy = np.copy(depth_map)

    min_d = np.min(depth_map_copy)
    max_d = np.max(depth_map_copy)
    depth_map_copy -= min_d
    depth_map_copy /= (max_d - min_d)
    depth_map_copy[depth_map_copy < min_truncate] = min_truncate
    depth_map_copy[depth_map_copy > max_truncate] = max_truncate

    # inverse show
    depth_map_copy -= min_truncate
    depth_map_copy /= (max_truncate - min_truncate)

    return depth_map_copy

# # Example:
# import matplotlib.pyplot as plt
# depth = load_depth_from_pgm('/mnt/Exp_1/For_Luwei/bear/depth/0145_1.pgm')
# save_depth_to_pgm(depth, '/mnt/Exp_1/For_Luwei/bear/depth/0145_1.pgm')
# plt.imshow(depth, cmap='jet')
# plt.show()
