from pathlib import Path
import trimesh
import numpy as np
import implicit_waterproofing as iw
import multiprocessing as mp
from multiprocessing import Pool
import argparse
from functools import partial
import traceback


def boundary_sampling(path: Path, sigma: float, sample_num: int):
    try:
        off_path = path
        out_file = path.parent / 'boundary_{}_samples.npz'.format(args.sigma)
        if out_file.exists():
            return

        mesh = trimesh.load(off_path)
        points = mesh.sample(sample_num)

        boundary_points = points + sigma * np.random.randn(sample_num, 3)
        grid_coords = boundary_points.copy()
        grid_coords[:, 0], grid_coords[:, 2] = boundary_points[:, 2], boundary_points[:, 0]

        grid_coords = 2 * grid_coords

        occupancies = iw.implicit_waterproofing(mesh, boundary_points)[0]

        np.savez(out_file, points=boundary_points, occupancies=occupancies, grid_coords=grid_coords)
        print('Finished {}'.format(path))
    except:
        print('Error with {}: {}'.format(path, traceback.format_exc()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run boundary sampling')
    parser.add_argument('-sigma', type=float)
    parser.add_argument('-root', required=True)
    args = parser.parse_args()

    sample_num = 100000

    paths = Path(args.root).glob('./*/*/*_scaled.off')
    p = Pool(mp.cpu_count())
    p.map(partial(boundary_sampling, sample_num=sample_num, sigma=args.sigma), paths)
