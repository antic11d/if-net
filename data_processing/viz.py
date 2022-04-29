import argparse
import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt


o3d_label_colors = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


def custom_draw_geometries(geom_arr):
    def capture_image(vis):
        image = vis.capture_screen_float_buffer()
        plt.imshow(np.asarray(image))
        plt.show()
        return False

    key_to_callback = {}
    key_to_callback[ord(".")] = capture_image
    o3d.visualization.draw_geometries_with_key_callbacks(geom_arr, key_to_callback)


def get_o3d_pointcloud(pc, color: np.ndarray = None, normal: np.ndarray = None):
    if isinstance(pc, o3d.geometry.PointCloud):
        if pc.has_normals() and normal is None:
            normal = np.asarray(pc.normals)
        if pc.has_colors() and color is None:
            color = np.asarray(pc.colors)
        pc = np.asarray(pc.points)

    assert (
        pc.shape[1] == 3 and len(pc.shape) == 2
    ), f"Point cloud is of size {pc.shape} and cannot be displayed!"
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pc)
    if color is not None and color.size != 0:
        assert (
            color.shape[0] == pc.shape[0]
        ), f"Point and color must have same size {color.shape[0]}, {pc.shape[0]}"
        point_cloud.colors = o3d.utility.Vector3dVector(color)
    if normal is not None and normal.size != 0:
        point_cloud.normals = o3d.utility.Vector3dVector(normal)

    return point_cloud


def viz_pc(points: np.ndarray, colors: np.ndarray = None):
    pc = get_o3d_pointcloud(points, colors)

    custom_draw_geometries([pc])


def labels_to_colors(labels: np.ndarray):
    return o3d_label_colors[labels]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', required=True)
    args = parser.parse_args()

    data = np.load(args.p)
    points = data['points']
    colors = labels_to_colors(data['labels'])

    viz_pc(points, colors)

