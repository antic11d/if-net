import trimesh
import numpy as np
import multiprocessing as mp
from multiprocessing import Pool
from pathlib import Path
from functools import partial
import argparse


def create_pc_off(path: Path, res: int, num_points: int):
    pc_path = path / 'voxelized_point_cloud_{}res_{}points.npz'.format(res, num_points)
    off_path = path / 'voxelized_point_cloud_{}res_{}points.off'.format(res, num_points)

    pc = np.load(pc_path)['point_cloud']

    trimesh.Trimesh(vertices=pc, faces=[]).export(off_path)
    print('Finished: {}'.format(path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create off visualization from point cloud.')

    parser.add_argument('-res', type=int)
    parser.add_argument('-num_points', type=int)
    parser.add_argument('-root', required=True)
    args = parser.parse_args()

    root = Path(args.root)

    p = Pool(mp.cpu_count())
    p.map(
        partial(create_pc_off, res=args.res, num_points=args.num_points), root.glob('./*/*'),
    )
