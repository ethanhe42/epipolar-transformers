import numpy as np

class Heatmapcreator():
    def __init__(self, output_size=(256, 256), sigma=10, downsample=4):
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        self.sigma = sigma * 2**.5
        self.downsample = downsample
        self.offset = downsample / 2.0 - 0.5
        self.grid = np.mgrid[0:self.output_size[0], 0:self.output_size[1]].astype(np.float32)
        self.grid = self.grid.reshape(1, 2, self.output_size[0], self.output_size[1])
        self.grid = (self.grid * self.downsample + self.offset) / self.sigma

    def get(self, coords_uv, valid_vec=None):
        """ Creates a map of size (output_shape[0], output_shape[1]) at (center[0], center[1])
            with variance sigma for multiple coordinates.
            coords_uvvis: Nx3
        """
        #cond_1_in = (coords_uv[:, 0] < self.output_size[0]-1) & (coords_uv[:, 0] > 0)
        #cond_2_in = (coords_uv[:, 1] < self.output_size[1]-1) & (coords_uv[:, 1] > 0)
        #cond= cond_1_in & cond_2_in
        #if valid_vec is not None:
        #    cond = (valid_vec.astype(float) > 0.5) & cond

        x = coords_uv[:, 1::-1].reshape(-1, 2, 1, 1) / self.sigma - self.grid
        scoremap = np.einsum('ijkl,ijkl->ikl', x, x)
        np.clip(scoremap, 0, 4.60517019, out=scoremap)
        np.exp(-scoremap, out=scoremap)
        if valid_vec is not None:
            scoremap[~valid_vec] = 0.
        #scoremap[~cond] = 0.

        return scoremap

def create_multiple_gaussian_map(coords_uv, output_size=(256, 256), sigma=25.0, valid_vec=None, grid=None):
    """ Creates a map of size (output_shape[0], output_shape[1]) at (center[0], center[1])
        with variance sigma for multiple coordinates.
        coords_uv: Nx2
    """
    assert len(output_size) == 2
    cond_1_in = (coords_uv[:, 0] < output_size[0]-1) & (coords_uv[:, 0] > 0)
    cond_2_in = (coords_uv[:, 1] < output_size[1]-1) & (coords_uv[:, 1] > 0)
    cond= cond_1_in & cond_2_in
    if valid_vec is not None:
        cond = (valid_vec.astype(float) > 0.5) & cond

    ## create meshgrid
    #x_range = np.arange(output_size[0])[:, None, None].astype(float)
    #y_range = np.arange(output_size[1])[None, :, None].astype(float)
    #X_b = np.tile(x_range, [1, output_size[1], coords_uv.shape[0]])
    #Y_b = np.tile(y_range, [output_size[0], 1, coords_uv.shape[0]])
    #X_b -= coords_uv[:, 0]
    #Y_b -= coords_uv[:, 1]
    #dist = X_b**2 + Y_b**2
    #before_exp = -dist / sigma**2

    if grid is None:
        grid = np.mgrid[0:output_size[0], 0:output_size[1]].astype(np.float32).reshape(1, 2, output_size[0], output_size[1]) / sigma
    x = coords_uv[:, :2].reshape(-1, 2, 1, 1) / sigma - grid
    scoremap = np.einsum('ijkl,ijkl->ikl', x, x)
    np.clip(scoremap, 0, 4.60517019, out=scoremap)
    np.exp(-scoremap, out=scoremap)
    scoremap[~cond] = 0.

    #before_exp = np.fromfunction(
    #        lambda pid, x, y: -((x - coords_uv[pid, 0])**2 + (y - coords_uv[pid, 1])**2)/ sigma**2,# / 2.  
    #        (coords_uv.shape[0], output_size[0], output_size[1]), 
    #        dtype=int
    #        )
    #idx = before_exp <= -4.60517019
    #before_exp[idx] = 0.
    #scoremap = np.exp(before_exp) * cond[:, None, None]
    #scoremap[idx] = 0.

    #scoremap = np.transpose(scoremap, (2, 0, 1))

    return scoremap

def xyxy_to_xywh(xyxy_box):
    xmin, ymin, xmax, ymax = xyxy_box
    xywh_box = (xmin, ymin, xmax - xmin + 1, ymax - ymin + 1)
    return xywh_box

def ccwh_to_xyxy(ccwh_box):
    cx, cy, w, h = ccwh_box
    return (
        int(cx - w / 2),
        int(cy - h / 2),
        int(cx + w / 2),
        int(cy + h / 2),
        )

def clip_to_image(xyxy_box, size):
    if len(xyxy_box) == 2:
        xmin, ymin = xyxy_box
        return (
            np.maximum(np.minimum(xmin, size[0] - 1), 0),
            np.maximum(np.minimum(ymin, size[1] - 1), 0),
            )

    xmin, ymin, xmax, ymax = xyxy_box
    return (
        np.maximum(np.minimum(xmin, size[0] - 1), 0),
        np.maximum(np.minimum(ymin, size[1] - 1), 0),
        np.maximum(np.minimum(xmax, size[0] - 1), 0),
        np.maximum(np.minimum(ymax, size[1] - 1), 0),
        )


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    h = Heatmapcreator((100, 50))
    coord = np.array([[ 100, 50], [200, 40]])
    plt.imshow(h.get(coord).sum(0))
    plt.show()
    

