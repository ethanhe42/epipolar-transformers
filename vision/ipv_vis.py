import numpy as np
import ipyvolume as ipv

def ipv_prepare(ipv, dark_background=True):
    ipv.clear()
    if dark_background is True:
        ipv.style.set_style_dark()


def ipv_draw_point_cloud(ipv, pts, colors, pt_size=10):
    pts = pts.reshape((-1, 3))
    colors = colors.reshape((-1, 3))
    assert colors.shape[0] == pts.shape[0]
    ipv.scatter(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], color=colors.reshape(-1, 3), marker='point_2d', size=pt_size)


def ipv_draw_pose_3d(ipv, R, t, color='blue', camera_scale=0.15, draw_axis_indicate=True):

    # camera obj
    cam_points = np.array([[0, 0, 0],
                           [-1, -1, 1.5],
                           [1, -1, 1.5],
                           [1, 1, 1.5],
                           [-1, 1, 1.5]])
    # axis indicators
    axis_points = np.array([[-0.5, 1, 1.5],
                            [0.5, 1, 1.5],
                            [0, 1.2, 1.5],
                            [1, -0.5, 1.5],
                            [1, 0.5, 1.5],
                            [1.2, 0, 1.5]])
    # transform camera objs...
    cam_points = (camera_scale * cam_points - t).dot(R)
    axis_points = (camera_scale * axis_points - t).dot(R)

    x = cam_points[:, 0]
    y = cam_points[:, 1]
    z = cam_points[:, 2]
    cam_wire_draw_order = np.asarray([(0, 1, 4, 0),  # left
                                      (0, 3, 2, 0),  # right
                                      (0, 4, 3, 0),  # top
                                      (0, 2, 1, 0)])  # bottom
    x = np.take(x, cam_wire_draw_order)
    y = np.take(y, cam_wire_draw_order)
    z = np.take(z, cam_wire_draw_order)

    axis_triangles = np.asarray([(3, 5, 4,),  # x-axis indicator
                                 (0, 1, 2)])  # y-axis indicator

    mesh = ipv.plot_wireframe(x, y, z, color=color, wrapx=True)
    mesh2 = ipv.plot_trisurf(x=axis_points[:, 0], y=axis_points[:, 1], z=axis_points[:, 2], triangles=axis_triangles,
                             color=color)

