from pathlib import Path
import trimesh
import numpy as np
import implicit_waterproofing as iw
import multiprocessing as mp
from multiprocessing import Pool
import argparse
from functools import partial
import traceback
from mesh_utils import get_labels, find_closest_vertices


def boundary_sampling(path: Path, sigma: float, sample_num: int, save_off: bool = True):
    try:
        root_dir = path.parent
        off_path = path
        out_file = root_dir / 'boundary_{}_samples.npz'.format(sigma)
        if out_file.exists():
            return

        mesh = trimesh.load(off_path)
        points = mesh.sample(sample_num)

        boundary_points = points + sigma * np.random.randn(sample_num, 3)
        grid_coords = boundary_points.copy()
        grid_coords[:, 0], grid_coords[:, 2] = boundary_points[:, 2], boundary_points[:, 0]
        grid_coords = 2 * grid_coords

        # Get segmentation labels for boundary points
        segm_labels = get_labels(root_dir)
        clean_scan = trimesh.load(root_dir / 'cleaned_scan.obj')

        _, idxs = find_closest_vertices(
            boundary_points.astype(np.float32), clean_scan.vertices.astype(np.float32)
        )
        labels = segm_labels[idxs]

        # Get occupancies
        occupancies = iw.implicit_waterproofing(mesh, boundary_points)[0]

        np.savez(
            out_file,
            points=boundary_points,
            labels=labels,
            occupancies=occupancies,
            grid_coords=grid_coords,
        )
        if save_off:
            save_path = out_file.parent / f'boundary_{sigma}.off'
            print(f'\tSaving at {save_path}')
            trimesh.Trimesh(vertices=boundary_points, faces=[]).export(save_path)

        print('Finished {}'.format(path))
    except:
        print('Error with {}: {}'.format(path, traceback.format_exc()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run boundary sampling')
    parser.add_argument('-sigma', type=float)
    parser.add_argument('-root', required=True)
    parser.add_argument('-debug', action='store_true', default=False)
    args = parser.parse_args()

    sample_num = 100000

    paths = Path(args.root).glob('./*/*/*_scaled.off')
    # use_cpu = mp.cpu_count() if not args.debug else 1
    # FAISS is struggling when using the whole pool of cpus.
    use_cpu = 1
    print(f'Running in pool of size {use_cpu}')

    p = Pool(use_cpu)
    p.map(partial(boundary_sampling, sample_num=sample_num, sigma=args.sigma), paths)
