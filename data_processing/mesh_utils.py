"""Code adapted from: https://github.com/garvita-tiwari/sizer"""
from pathlib import Path
import numpy as np
from os.path import split
from psbody.mesh import Mesh
import cv2
import faiss

label_colours = [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255)]
garms = [0, 1, 2, 3]


def find_closest_vertices(points: np.ndarray, vertices: np.ndarray, use_gpu: bool = False):
    d = 3

    index = faiss.IndexFlatL2(d)
    if use_gpu:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)

    index.add(vertices)
    D, I = index.search(points, 1)

    return D, I.squeeze(-1)


def get_labels(scan_dir: Path):
    scan_raw = read_obj(scan_dir / 'model_0.8.obj')
    scan_mesh = Mesh(v=scan_raw.v, f=scan_raw.f)
    scan_mesh.vt = scan_raw.vt
    scan_mesh.ft = scan_raw.ft

    seg_path = str(scan_dir / 'seg_0.8_new.png')
    seg_labels = _load_labels(cv2.imread(seg_path), scan_mesh.vt, scan_mesh.ft, scan_mesh.f)

    bg_vid = np.where(seg_labels != 0)[0]
    _, _, _, vertex_ids = get_submesh(scan_mesh.v, scan_mesh.f, bg_vid)

    return seg_labels[vertex_ids]


def _load_labels(model_seg, vth, fth, fh):
    _, tex_indices_ = np.unique(fh.ravel(), return_index=True)
    tex_indices = np.array(vth[fth.ravel()[tex_indices_]] * model_seg.shape[:2], dtype='int32')
    # this is for sizer and frl dataset, where vt max is 1 and hence last index = img size
    labels = [tuple(model_seg[::-1, ...][y - 1, x - 1]) for x, y in tex_indices]
    dic = {}
    for n, l in enumerate(label_colours):
        dic[l] = garms[n]
    model_label = np.array([dic[x] for x in labels])
    return model_label


# Helper class
class Minimal(object):
    def __init__(self, **kwargs):
        self.__dict__ = kwargs


def get_submesh(verts, faces, verts_retained=None, faces_retained=None, min_vert_in_face=2):
    '''
        Given a mesh, create a (smaller) submesh
        indicate faces or verts to retain as indices or boolean

        @return new_verts: the new array of 3D vertices
                new_faces: the new array of faces
                bool_faces: the faces indices wrt the input mesh
                vetex_ids: the vertex_ids wrt the input mesh
        '''

    if verts_retained is not None:
        # Transform indices into bool array
        if verts_retained.dtype != 'bool':
            vert_mask = np.zeros(len(verts), dtype=bool)
            vert_mask[verts_retained] = True
        else:
            vert_mask = verts_retained

        # Faces with at least min_vert_in_face vertices
        bool_faces = np.sum(vert_mask[faces.ravel()].reshape(-1, 3), axis=1) > min_vert_in_face

    elif faces_retained is not None:
        # Transform indices into bool array
        if faces_retained.dtype != 'bool':
            bool_faces = np.zeros(len(faces_retained), dtype=bool)
        else:
            bool_faces = faces_retained

    new_faces = faces[bool_faces]
    # just in case additional vertices are added
    vertex_ids = list(set(new_faces.ravel()))

    oldtonew = -1 * np.ones([len(verts)])
    oldtonew[vertex_ids] = range(0, len(vertex_ids))

    new_verts = verts[vertex_ids]
    new_faces = oldtonew[new_faces].astype('int32')

    return new_verts, new_faces, bool_faces, vertex_ids


def read_obj(filename):
    ###Thanks to Thiemo for this code
    obj_directory = split(filename)[0]
    lines = open(filename).read().split('\n')

    d = {'v': [], 'vn': [], 'f': [], 'vt': [], 'ft': []}

    mtls = {}
    for line in lines:
        line = line.split()
        if len(line) < 2:
            continue

        key = line[0]
        values = line[1:]

        if key == 'v':
            d['v'].append([np.array([float(v) for v in values[:3]])])
        elif key == 'f':
            spl = [l.split('/') for l in values]
            d['f'].append([np.array([int(l[0]) - 1 for l in spl[:3]], dtype=np.int32)])
            if len(spl[0]) > 1 and spl[1] and len(spl[0][1]) > 0 and 'ft' in d:
                d['ft'].append([np.array([int(l[1]) - 1 for l in spl[:3]])])

            # TOO: redirect to actual vert normals?
            # if len(line[0]) > 2 and line[0][2]:
            #    d['fn'].append([np.concatenate([l[2] for l in spl[:3]])])
        elif key == 'vn':
            d['vn'].append([np.array([float(v) for v in values])])
        elif key == 'vt':
            d['vt'].append([np.array([float(v) for v in values])])

    for k, v in d.items():
        if k in ['v', 'vn', 'f', 'vt', 'ft']:
            if v:
                d[k] = np.vstack(v)
            else:
                del d[k]
        else:
            d[k] = v

    result = Minimal(**d)

    return result
