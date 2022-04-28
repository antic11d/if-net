import multiprocessing as mp
from multiprocessing import Pool
import argparse
import os
from pathlib import Path
import shutil
import numpy as np
from functools import partial


def filter(path: Path, delete: bool, file: str):
    if not (path / file).exists():
        print('Should remove: {}'.format(path, file))
        if delete:
            print('\tRemoving...')
            shutil.rmtree('{}'.format(path))


def update_split():
    split = np.load('shapenet/split.npz')
    split_dict = {}
    for set in ['train', 'test', 'val']:
        filterd_set = split[set].copy()
        for path in split[set]:
            if not os.path.exists('shapenet/data/' + path):
                print('Filtered: ' + path)
                filterd_set = np.delete(filterd_set, np.where(filterd_set == path))
        split_dict[set] = filterd_set

    np.savez('shapenet/split.npz', **split_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Filter objects if preprocessing failed.')
    parser.add_argument('-file', type=str)
    parser.add_argument('-delete', action='store_true', default=False)
    parser.add_argument('-update_split', action='store_true', default=False)
    parser.add_argument('-root', required=True)
    args = parser.parse_args()

    paths = [p for p in Path(args.root).glob('./*/*') if p.is_dir()]
    p = Pool(mp.cpu_count())
    p.map(partial(filter, delete=args.delete, file=args.file), paths)

    if args.update_split:
        update_split()
