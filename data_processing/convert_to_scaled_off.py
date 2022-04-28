import os
import multiprocessing as mp
from multiprocessing import Pool
import trimesh
from pathlib import Path
from argparse import ArgumentParser


def to_off(path: Path):
    if (path / 'isosurf.off').exists():
        return

    input_file = path
    output_file = path.parent / 'isosurf.off'

    cmd = 'meshlabserver -i {} -o {}'.format(input_file, output_file)
    # if you run this script on a server: comment out above line and uncomment the next line
    # cmd = 'xvfb-run -a -s "-screen 0 800x600x24" meshlabserver -i {} -o {}'.format(input_file,output_file)
    os.system(cmd)


def scale(path: Path):
    if (path / 'isosurf_scaled.off').exists():
        print(f'{path}/isosurf_scaled.off already exists!')
        return

    try:
        mesh = trimesh.load(path, process=False)
        total_size = (mesh.bounds[1] - mesh.bounds[0]).max()
        centers = (mesh.bounds[1] + mesh.bounds[0]) / 2

        mesh.apply_translation(-centers)
        mesh.apply_scale(1 / total_size)
        save_path = path.parent / 'isosurf_scaled.off'
        mesh.export(save_path)
    except:
        print('Error with {}'.format(path))
    print('Finished {}'.format(path))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-path', required=True)
    args = parser.parse_args()

    path = Path(args.path)

    p = Pool(mp.cpu_count())
    p.map(to_off, path.glob('./*/*.obj'))

    p = Pool(mp.cpu_count())
    p.map(scale, path.glob('./*/*.off'))

