#!/usr/bin/env python3

from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from numpy_utils import calculate_tangent_angles

from  grid_sample import numpy_grid_sample

__all__ = ["convert2batches", "run"]


def single_list2horizon(cube: List[np.ndarray]) -> np.ndarray:
    _, _, w = cube[0].shape
    assert len(cube) == 6
    assert sum(face.shape[-1] == w for face in cube) == 6
    return np.concatenate(cube, axis=-1)


def dice2horizon(dices: np.ndarray) -> np.ndarray:
    assert len(dices.shape) == 4
    w = dices.shape[-2] // 3
    assert dices.shape[-2] == w * 3 and dices.shape[-1] == w * 4

    # create a (b, c, h, w) horizon array
    horizons = np.empty((*dices.shape[0:2], w, w * 6), dtype=dices.dtype)

    # Order: F R B L U D
    sxy = [(1, 1), (2, 1), (3, 1), (0, 1), (1, 0), (1, 2)]
    for i, (sx, sy) in enumerate(sxy):
        horizons[..., i * w : (i + 1) * w] = dices[
            ..., sy * w : (sy + 1) * w, sx * w : (sx + 1) * w
        ]
    return horizons


def dict2horizon(dicts: List[Dict[str, np.ndarray]]) -> np.ndarray:
    face_key = ("F", "R", "B", "L", "U", "D")
    c, _, w = dicts[0][face_key[0]].shape
    dtype = dicts[0][face_key[0]].dtype
    horizons = np.empty((len(dicts), c, w, w * 6), dtype=dtype)
    for b, cube in enumerate(dicts):
        horizons[b, ...] = single_list2horizon([cube[k] for k in face_key])
    return horizons


def list2horizon(lists: List[List[np.ndarray]]) -> np.ndarray:
    assert len(lists[0][0].shape) == 3
    c, w, _ = lists[0][0].shape
    dtype = lists[0][0].dtype
    horizons = np.empty((len(lists), c, w, w * 6), dtype=dtype)
    for b, cube in enumerate(lists):
        horizons[b, ...] = single_list2horizon(cube)
    return horizons


def convert2batches(
    cubemap: Union[
        np.ndarray,
        List[np.ndarray],
        List[List[np.ndarray]],
        Dict[str, np.ndarray],
        List[Dict[str, np.ndarray]],
    ],
    cube_format: str,
) -> np.ndarray:
    """Converts supported cubemap formats to horizon

    params:
    - cubemap
    - cube_format (str): ('horizon', 'dice', 'dict', 'list')

    return:
    - horizon (np.ndarray)

    """

    # FIXME: better typing for mypy...

    if cube_format == "list":
        assert isinstance(
            cubemap, list
        ), f"ERR: cubemap {cube_format} needs to be a list"
        if isinstance(cubemap[0], np.ndarray):
            # single
            cubemap = [cubemap]
        cubemap = list2batch(cubemap)  # type: ignore
    elif cube_format == "dict":
        if isinstance(cubemap, dict):
            cubemap = [cubemap]
        assert isinstance(cubemap, list)
        assert isinstance(
            cubemap[0], dict
        ), f"ERR: cubemap {cube_format} needs to have dict inside the list"
        cubemap = dict2horizon(cubemap)  # type: ignore
    else:
        raise ValueError(f"ERR: {cube_format} is not supported")

    assert (
        len(cubemap.shape) == 4
    ), f"ERR: cubemap needs to be 4 dim, but got {cubemap.shape}"

    return cubemap


def _equirect_facetype(h: int, w: int) -> np.ndarray:
    """0F 1R 2B 3L 4U 5D"""

    int_dtype = np.dtype(np.int64)

    tp = np.roll(
        np.arange(4).repeat(w // 4)[None, :].repeat(h, 0), 3 * w // 8, 1
    )

    # Prepare ceil mask
    mask = np.zeros((h, w // 4), np.bool)
    idx = np.linspace(-np.pi, np.pi, w // 4) / 4
    idx = h // 2 - np.around(np.arctan(np.cos(idx)) * h / np.pi)
    idx = idx.astype(int_dtype)
    for i, j in enumerate(idx):
        mask[:j, i] = 1
    mask = np.roll(np.concatenate([mask] * 4, 1), 3 * w // 8, 1)

    tp[mask] = 4
    tp[np.flip(mask, 0)] = 5

    return tp.astype(int_dtype)


def create_equi_grid(
    h_out: int,
    w_out: int,
    w_face: int,
    batch: int,
    dtype: np.dtype = np.dtype(np.float32),
) -> np.ndarray:
    theta = np.linspace(-np.pi, np.pi, num=w_out, dtype=dtype)
    phi = np.linspace(np.pi, -np.pi, num=h_out, dtype=dtype) / 2
    theta, phi = np.meshgrid(theta, phi)

    # Get face id to each pixel: 0F 1R 2B 3L 4U 5D
    tp = _equirect_facetype(h_out, w_out)

    # xy coordinate map
    coor_x = np.zeros((h_out, w_out), dtype=dtype)
    coor_y = np.zeros((h_out, w_out), dtype=dtype)

    for i in range(6):
        mask = tp == i

        if i < 4:
            coor_x[mask] = 0.5 * np.tan(theta[mask] - np.pi * i / 2)
            coor_y[mask] = (
                -0.5 * np.tan(phi[mask]) / np.cos(theta[mask] - np.pi * i / 2)
            )
        elif i == 4:
            c = 0.5 * np.tan(np.pi / 2 - phi[mask])
            coor_x[mask] = c * np.sin(theta[mask])
            coor_y[mask] = c * np.cos(theta[mask])
        elif i == 5:
            c = 0.5 * np.tan(np.pi / 2 - np.abs(phi[mask]))
            coor_x[mask] = c * np.sin(theta[mask])
            coor_y[mask] = -c * np.cos(theta[mask])

    # Final renormalize
    coor_x = np.clip(np.clip(coor_x + 0.5, 0, 1) * w_face, 0, w_face - 1)
    coor_y = np.clip(np.clip(coor_y + 0.5, 0, 1) * w_face, 0, w_face - 1)

    # change x axis of the x coordinate map
    for i in range(6):
        mask = tp == i
        coor_x[mask] = coor_x[mask] + w_face * i

    grid = np.stack((coor_y, coor_x), axis=0)
    grid = np.concatenate([grid[np.newaxis, ...]] * batch)
    return grid


# -----------------------------------------------------
def ceil_max(a: np.array):
    return np.sign(a)*np.ceil(np.abs(a))

def rodrigues(rot_vector: np.ndarray):
    if rot_vector.shape == (3,):
        theta = np.linalg.norm(rot_vector)
        i = np.eye(3)
        if theta < 1e-9:
            return i
        r = rot_vector / theta
        rr = np.tile(r, (3,1))*np.tile(r, (3,1)).T
        rmap = np.array([
            [0, -r[2], r[1]],
            [r[2], 0, -r[0]],
            [-r[1], r[0], 0]
        ])
        return (np.cos(theta)*i + (1-np.cos(theta))*rr + np.sin(theta)*rmap).astype(np.float32)
    else:
        R = rot_vector
        r = np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])
        s = np.sqrt(np.sum(r**2)/4)
        c = (np.sum(np.eye(3)*R)-1)/2
        c = np.clip(c, -1, 1)
        theta_ = np.arccos(c)
        
        if c > 0:
            return np.zeros(3, np.float32)
        
        if s < 1e-5:
            t = (R[0,0]+1)/2
            r[0] = np.sqrt(np.max([t, 0]))
            t = (R[1,1]+1)/2
            r[1] = np.sqrt(np.max([t,0]))*ceil_max(R[0,1])
            t = (R[2,2]+1)/2
            r[2] = np.sqrt(np.max([t,0]))*ceil_max(R[0,2])
            abs_r = np.abs(r)
            abs_r -= abs_r[0]
            if (abs_r[1] > 0) and (abs_r[2] > 0) and (R[1,2] > 0 != r[1]*r[2]>0):
                r[2] = -r[2]
            theta_ /= np.linalg.norm(r)
            r *= theta_
        else:
            vth = 1/(2*s) * theta_
            r *= vth
            
        return r.reshape(3,1).astype(np.float32)


def get_equirec(
    img: np.ndarray, 
    fov_x: float,
    theta: int,
    phi: int,
    height: int,
    width: int,
):
    _img = img
    _height, _width = _img.shape[1:]
    wFOV = fov_x
    THETA = theta
    PHI = phi
    hFOV = float(_height) / _width * fov_x

    w_len = np.tan(np.radians(wFOV / 2.0))
    h_len = np.tan(np.radians(hFOV / 2.0))
    #
    # THETA is left/right angle, PHI is up/down angle, both in degree
    #

    x,y = np.meshgrid(np.linspace(-180, 180,width),np.linspace(90,-90,height))
    
    x_map = np.cos(np.radians(x)) * np.cos(np.radians(y))
    y_map = np.sin(np.radians(x)) * np.cos(np.radians(y))
    z_map = np.sin(np.radians(y))

    xyz = np.stack((x_map,y_map,z_map),axis=2)

    y_axis = np.array([0.0, 1.0, 0.0], np.float32)
    z_axis = np.array([0.0, 0.0, 1.0], np.float32)
    R1 = rodrigues(z_axis * np.radians(THETA))
    R2 = rodrigues(np.dot(R1, y_axis) * np.radians(-PHI))

    R1 = np.linalg.inv(R1)
    R2 = np.linalg.inv(R2)

    xyz = xyz.reshape([height * width, 3]).T
    xyz = np.dot(R2, xyz)
    xyz = np.dot(R1, xyz).T

    xyz = xyz.reshape([height , width, 3])
    inverse_mask = np.where(xyz[:,:,0]>0,1,0)

    xyz[:,:] = xyz[:,:]/np.repeat(xyz[:,:,0][:, :, np.newaxis], 3, axis=2)
    
    
    lon_map = np.where((-w_len<xyz[:,:,1])&(xyz[:,:,1]<w_len)&(-h_len<xyz[:,:,2])
                &(xyz[:,:,2]<h_len),(xyz[:,:,1]+w_len)/2/w_len*_width,0)
    lat_map = np.where((-w_len<xyz[:,:,1])&(xyz[:,:,1]<w_len)&(-h_len<xyz[:,:,2])
                &(xyz[:,:,2]<h_len),(-xyz[:,:,2]+h_len)/2/h_len*_height,0)
    mask = np.where((-w_len<xyz[:,:,1])&(xyz[:,:,1]<w_len)&(-h_len<xyz[:,:,2])
                &(xyz[:,:,2]<h_len),1,0)

    # TODO: FIX shapes and pipeline in sourcecode

    dtype = np.float32
    out = np.empty((1, 3, height, width), dtype=dtype)
    grid = np.stack((lat_map, lon_map), axis=0)
    grid = np.concatenate([grid[np.newaxis, ...]] * 1)
    imgt = np.concatenate([_img[np.newaxis, ...]] * 1)
    out = numpy_grid_sample(imgt, grid, out, "bilinear")
    out = out.squeeze()
    
    mask = mask * inverse_mask
    mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    mask = np.transpose(mask, (2,0,1))
    persp = out * mask
    
    return persp, mask


def run(
    icomaps: List[np.ndarray],
    height: int,
    width: int,
    fov_x: float,
    mode: str,
    override_func: Optional[Callable[[], Any]] = None,
) -> np.ndarray:
    """Run Cube2Equi

    params:
    - icomaps (np.ndarray)
    - height, widht (int): output equirectangular image's size
    - mode (str)

    return:
    - equi (np.ndarray)

    NOTE: we assume that the input `horizon` is a 4 dim array

    """

    assert (
        len(icomaps) >= 1 and len(icomaps[0].shape)==4
    ), f"ERR: `icomaps` should be 4-dim (b, fn, c, h, w), but got {icomaps.shape}"

    icomaps_dtype = icomaps[0].dtype
    assert icomaps_dtype in (np.uint8, np.float32, np.float64), (
        f"ERR: input horizon has dtype of {icomaps_dtype}\n"
        f"which is incompatible: try {(np.uint8, np.float32, np.float64)}"
    )

    # NOTE: we don't want to use uint8 as output array's dtype yet since
    # floating point calculations (matmul, etc) are faster
    # NOTE: we are also assuming that uint8 is in range of 0-255 (obviously)
    # and float is in range of 0.0-1.0; later we will refine it
    # NOTE: for the sake of consistency, we will try to use the same dtype as horizon
    dtype = (
        np.dtype(np.float32)
        if icomaps_dtype == np.dtype(np.uint8)
        else icomaps_dtype
    )
    assert dtype in (np.float32, np.float64), (
        f"ERR: argument `dtype` is {dtype} which is incompatible:\n"
        f"try {(np.float32, np.float64)}"
    )

    bs,c = len(icomaps), icomaps[0].shape[1]

    # initialize output equi
    out_batch = np.empty((bs, c, height, width), dtype=icomaps_dtype)
    subdivision_levels = [int(np.log(icomap.shape[0]/20)/np.log(4)) for icomap in icomaps]
    angles = calculate_tangent_angles(subdivision_levels)

    for bn, (imgs, angle) in enumerate(zip(icomaps, angles)):
        angle *= -1*180/np.pi
        out = np.empty((c, height, width), dtype=dtype)
        merge_image = np.zeros((c,height,width))
        merge_mask = np.zeros((c,height,width))
        for img,[T,P] in zip(imgs, angle):
            img, mask = get_equirec(img,fov_x,T,P,height, width)
            merge_image += img
            merge_mask += mask
        merge_mask = np.where(merge_mask==0,1,merge_mask)
        out = np.divide(merge_image,merge_mask)

        out = (
            out.astype(icomaps_dtype)
            if icomaps_dtype == np.dtype(np.uint8)
            else np.clip(out, 0.0, 1.0)
        )
        out_batch[bn] = out

    return out_batch
