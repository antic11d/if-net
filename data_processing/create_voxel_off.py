from voxels import VoxelGrid
import numpy as np
import multiprocessing as mp
from multiprocessing import Pool
import argparse
from pathlib import Path
from functools import partial


def create_voxel_off(path: Path, res: int, min: float, max: float):
    voxel_path = path / f'voxelization_{res}.npy'
    off_path = path / f'voxelization_{res}.off'

    if unpackbits:
        occ = np.unpackbits(np.load(voxel_path))
        voxels = np.reshape(occ, (res,) * 3)
    else:
        voxels = np.reshape(np.load(voxel_path)['occupancies'], (res,) * 3)

    loc = ((min + max) / 2,) * 3
    scale = max - min

    VoxelGrid(voxels, loc, scale).to_mesh().export(off_path)
    print('Finished: {}'.format(path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run voxalization to off')
    parser.add_argument('-res', type=int)
    parser.add_argument('-root', required=True)
    args = parser.parse_args()

    root = Path(args.root)

    unpackbits = True
    res = args.res
    min = -0.5
    max = 0.5

    p = Pool(mp.cpu_count())
    p.map(partial(create_voxel_off, res=args.res, min=min, max=max), root.glob('./*/*'))

