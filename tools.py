import math
import numpy as np
import torch
import torch.nn.functional as F
import functools
import matplotlib.pyplot as plt
import cv2
import math
import scipy
from skimage import transform

import haya_ext
import haya_data

_mean = torch.tensor([0.485, 0.456, 0.406],
                     dtype=torch.float32).view(1, 3, 1, 1)
_std = torch.tensor([0.229, 0.224, 0.225],
                    dtype=torch.float32).view(1, 3, 1, 1)


def normalize_and_offset_images(im_u8):
    """ normalize images
    Inputs:
    - im_u8: uint8, b x 3 x h x w
    """
    return (im_u8.type(torch.float32) / 255.0 -
            _mean.to(im_u8.device)) / _std.to(im_u8.device)


def deoffset_images(nim):
    return nim * _std.to(nim.device) + _mean.to(nim.device)


def load_body_mesh():
    return haya_ext.read_trimesh_from_obj_file('./female_wholebody_low.obj',
                                               True)


def radians(degrees):
    return degrees * math.pi / 180.0


def dot(a, b, dim=-1):
    """batched dot product
    Inputs:
    - a, b: float, [batch x k]

    Returns:
    - prod: float, [batch]
    """
    return (a * b).sum(dim)


def normalize(x, dim):
    return x / torch.max(x.norm(None, dim=dim, keepdim=True), 1e-5)


def translate(v):
    """ translate
    Inputs:
    - v: float, [batch x 3]

    Returns:
    - mat: float, [batch x 4 x 4]
    """
    zeros_pl = torch.zeros_like(v[:, 0])
    ones_pl = torch.ones_like(v[:, 0])
    return torch.stack([
        ones_pl, zeros_pl, zeros_pl, v[:, 0], zeros_pl, ones_pl, zeros_pl,
        v[:, 1], zeros_pl, zeros_pl, ones_pl, v[:, 2], zeros_pl, zeros_pl,
        zeros_pl, ones_pl
    ], -1).view(-1, 4, 4)


def yaw_pitch_roll(yaw, pitch, roll):
    """ yaw_pitch_roll
    Inputs:
    - yaw, pitch, roll: float, [batch]

    Returns:
    - mat: float, [batch x 4 x 4]
    """
    zeros_pl = torch.zeros_like(yaw)
    ones_pl = torch.ones_like(pitch)

    tmp_ch = torch.cos(yaw)
    tmp_sh = torch.sin(yaw)
    tmp_cp = torch.cos(pitch)
    tmp_sp = torch.sin(pitch)
    tmp_cb = torch.cos(roll)
    tmp_sb = torch.sin(roll)

    return torch.stack([
        tmp_ch * tmp_cb + tmp_sh * tmp_sp * tmp_sb, tmp_sb * tmp_cp,
        -tmp_sh * tmp_cb + tmp_ch * tmp_sp * tmp_sb, zeros_pl,
        -tmp_ch * tmp_sb + tmp_sh * tmp_sp * tmp_cb, tmp_cb * tmp_cp,
        tmp_sb * tmp_sh + tmp_ch * tmp_sp * tmp_cb, zeros_pl, tmp_sh * tmp_cp,
        -tmp_sp, tmp_ch * tmp_cp, zeros_pl, zeros_pl, zeros_pl, zeros_pl,
        ones_pl
    ], -1).view(-1, 4, 4).permute(0, 2, 1)


def orientate4(angles):
    """oriente4
    Inputs:
    - angles: float, [batch x 3], (x, y, z)

    Returns:
    - mat: float, [batch x 4 x 4]
    """
    return yaw_pitch_roll(angles[:, 2], angles[:, 0], angles[:, 1])


def look_at_rh(eyes, centers, ups):
    """look at (rh)
    Inputs:
    - eyes, centers, ups: float, [batch x 3]

    Returns:
    - view_mat: float, [batch x 4 x 4]
    """
    f = normalize(centers - eyes, dim=1)
    s = normalize(torch.cross(f, ups, dim=1), dim=1)
    u = torch.cross(s, f, dim=1)

    zeros_pl = torch.zeros([eyes.size(0)],
                           dtype=eyes.dtype,
                           device=eyes.device)
    ones_pl = torch.ones([eyes.size(0)], dtype=eyes.dtype, device=eyes.device)

    return torch.stack([
        s[:, 0], s[:, 1], s[:, 2], zeros_pl, u[:, 0], u[:, 1], u[:, 2],
        zeros_pl, -f[:, 0], -f[:, 1], -f[:, 2], zeros_pl, -dot(s, eyes),
        -dot(u, eyes),
        dot(f, eyes), ones_pl
    ], -1).view(-1, 4, 4).permute(0, 2, 1)


def look_at_lh(eyes, centers, ups):
    """look at (lh)
    Inputs:
    - eyes, centers, ups: float, [batch x 3]

    Returns:
    - view_mat: float, [batch x 4 x 4]
    """
    f = normalize(centers - eyes, dim=1)
    s = normalize(torch.cross(ups, f, dim=1), dim=1)
    u = torch.cross(f, s, dim=1)

    zeros_pl = torch.zeros([eyes.size(0)],
                           dtype=eyes.dtype,
                           device=eyes.device)
    ones_pl = torch.ones([eyes.size(0)], dtype=eyes.dtype, device=eyes.device)

    return torch.stack([
        s[:, 0], s[:, 1], s[:, 2], zeros_pl, u[:, 0], u[:, 1], u[:, 2],
        zeros_pl, f[:, 0], f[:, 1], f[:, 2], zeros_pl, -dot(s, eyes),
        -dot(u, eyes), -dot(f, eyes), ones_pl
    ], -1).view(-1, 4, 4).permute(0, 2, 1)


look_at = look_at_rh


def perspective_rh_no(fovy, aspect, z_near, z_far):
    """ perspective (rh_no)
    Inputs:
    - fovy, aspect, z_near, z_far: [batch]

    Returns:
    - proj_mat: float, [batch x 4 x 4]
    """
    tan_half_fovy = torch.tan(fovy / 2.0)
    zeros_pl = torch.zeros_like(fovy)
    ones_pl = torch.ones_like(fovy)

    return torch.stack([
        1.0 / aspect / tan_half_fovy, zeros_pl, zeros_pl, zeros_pl, zeros_pl,
        1.0 / tan_half_fovy, zeros_pl, zeros_pl, zeros_pl, zeros_pl,
        -(z_far + z_near) / (z_far - z_near), -ones_pl, zeros_pl, zeros_pl,
        -2.0 * z_far * z_near / (z_far - z_near), zeros_pl
    ], -1).view(-1, 4, 4).permute(0, 2, 1)


def perspective_rh_zo(fovy, aspect, z_near, z_far):
    """ perspective (rh_zo)
    Inputs:
    - fovy, aspect, z_near, z_far: [batch]

    Returns:
    - proj_mat: float, [batch x 4 x 4]
    """
    tan_half_fovy = torch.tan(fovy / 2.0)
    zeros_pl = torch.zeros_like(fovy)
    ones_pl = torch.ones_like(fovy)

    return torch.stack([
        1.0 / aspect / tan_half_fovy, zeros_pl, zeros_pl, zeros_pl, zeros_pl,
        1.0 / tan_half_fovy, zeros_pl, zeros_pl, zeros_pl, zeros_pl, z_far /
        (z_near - z_far), -ones_pl, zeros_pl, zeros_pl, -z_far * z_near /
        (z_far - z_near), zeros_pl
    ], -1).view(-1, 4, 4).permute(0, 2, 1)


perspective = perspective_rh_no


def _get_hair10k_model_matrix(rotation_head, translation_head):
    srx, sry, srz = torch.split(torch.sin(rotation_head), 1, dim=1)
    crx, cry, crz = torch.split(torch.cos(rotation_head), 1, dim=1)
    tx, ty, tz = torch.split(translation_head, 1, dim=1)
    zeros_pl = torch.zeros_like(tx)
    ones_pl = torch.ones_like(tx)

    return torch.cat([
        cry * crz + srx * sry * srz, cry * srx * srz - crz * sry, crx * srz,
        tx + (17 * crz * sry) / 10 -
        (17 * cry * srx * srz) / 10, crx * sry, crx * cry, -srx, ty -
        (17 * crx * cry) / 10, crz * srx * sry - cry * srz,
        sry * srz + cry * crz * srx, crx * crz, tz - (17 * sry * srz) / 10 -
        (17 * cry * crz * srx) / 10, zeros_pl, zeros_pl, zeros_pl, ones_pl
    ],
                     dim=1).view(-1, 4, 4)


def _mul_hair10k_projection_view_matrix(model):
    mvp = torch.matmul(
        torch.tensor([[[4.2468, 0.0000, 0.0000, 0.0000],
                       [0.0000, 4.2468, 0.0000, 0.0000],
                       [0.0000, 0.0000, -1.2857, 0.5286],
                       [0.0000, 0.0000, -1.0000, 1.3000]]],
                     dtype=model.dtype,
                     device=model.device), model)
    return mvp


def compute_hair10k_mvp(translation_head, rotation_head):
    """ compute_hair10k_mvp
    Inputs:
    - translation_head: batch x 3
    - rotation_head: batch x 3
    Outputs:
    - mvp: batch x 4 x 4
    """
    return _mul_hair10k_projection_view_matrix(
        _get_hair10k_model_matrix(rotation_head, translation_head))


def generate_random_mvps(batchsize, device):
    # generate random trans_heads and rot_heads
    zeros = torch.zeros(batchsize, device=device)
    ones = torch.ones(batchsize, device=device)
    rot_yaw = torch.normal(mean=zeros, std=ones * math.pi / 4)
    rot_pitch = torch.normal(mean=zeros, std=ones * math.pi / 12)
    rot_roll = torch.normal(mean=zeros, std=ones * math.pi / 12)

    rot_heads = torch.stack([rot_roll, rot_pitch, rot_yaw], dim=1)
    trans_heads = torch.normal(mean=torch.zeros(batchsize, 3, device=device),
                               std=torch.ones(batchsize, 3, device=device) *
                               0.05)

    # compute mvps
    mvps = compute_hair10k_mvp(trans_heads, rot_heads)
    return mvps, rot_heads, trans_heads

def generate_specific_mvps(rot, tran):
    '''
    rot: B,3
    tran: B,3
    '''
    tran = torch.zeros_like(tran)
    mvps = compute_hair10k_mvp(tran, rot[:, inds])
    # dxdy, strand_mask, body_mask, *_ = render(mvps.cuda(), strands.cuda(),
    #                                                 body_mesh_vert_pos.cuda(),
    #                                                 body_mesh_face_inds.cuda(),
    #                                                 512,
    #                                                 0.5,
    #                                                 align_face=True,
    #                                                 target_face_scale=0.7)
    return mvps, rot, tran

def show_info2d(info2d):
    import matplotlib.pyplot as plt
    for i in range(info2d.size(0)):
        plt.subplot(121)
        plt.imshow(info2d[i, [0, 2, 3], :, :].abs().permute(1, 2,
                                                            0).cpu().data)
        plt.subplot(122)
        plt.imshow(info2d[i, [1, 2, 3], :, :].abs().permute(1, 2,
                                                            0).cpu().data)
        plt.show()


def batch_inverse(m):
    eye = m.new_ones(m.size(-1)).diag().expand_as(m)
    b_inv, _ = torch.gesv(eye, m)
    return b_inv


def transform_strands(mvp, strands):
    """
    Inputs:
    - mvp: float, batch x 4 x 4
    - strands: float, batch x 3 x sh x sw x sn
    Returns:
    - trans_strands: float, batch x 3 x sh x sw x sn
    """
    b, _, sh, sw, sn = strands.shape
    assert mvp.size(0) == b
    homo_strands = torch.cat([
        strands,
        torch.ones(
            [b, 1, sh, sw, sn], dtype=strands.dtype, device=strands.device)
    ], 1)  # b x 4 x sh x sw x sn
    homo_strands = homo_strands.view(b, 4, -1)
    homo_trans_strands = torch.matmul(mvp,
                                      homo_strands)  # b x 4 x (sh x sw sn)
    homo_trans_strands3 = homo_trans_strands[:, [3]]
    homo_trans_strands3 = torch.where(
        homo_trans_strands3 == 0,
        torch.ones_like(homo_trans_strands3) * 1e-5, homo_trans_strands3)
    trans_strands = homo_trans_strands[:, :3] / homo_trans_strands3
    trans_strands = trans_strands.view(b, 3, sh, sw, sn)
    return trans_strands


def inverse_transform_strands(mvp, strands):
    """
    Inputs:
    - mvp: float, batch x 4 x 4
    - strands: float, batch x 3 x sh x sw x sn
    Returns:
    - trans_strands: float, batch x 3 x sh x sw x sn
    """
    b, _, sh, sw, sn = strands.shape
    assert mvp.size(0) == b
    homo_strands = torch.cat([
        strands,
        torch.ones(
            [b, 1, sh, sw, sn], dtype=strands.dtype, device=strands.device)
    ], 1)  # b x 4 x sh x sw x sn
    homo_strands = homo_strands.view(b, 4, -1)
    homo_trans_strands, _ = torch.gesv(homo_strands,
                                       mvp)  # b x 4 x (sh x sw sn)
    homo_trans_strands3 = homo_trans_strands[:, [3]]
    homo_trans_strands3 = torch.where(
        homo_trans_strands3 == 0,
        torch.ones_like(homo_trans_strands3) * 1e-5, homo_trans_strands3)
    trans_strands = homo_trans_strands[:, :3] / homo_trans_strands3
    trans_strands = trans_strands.view(b, 3, sh, sw, sn)
    return trans_strands


def transform_mesh_verts(mvp, verts):
    """
    Inputs:
    - mvp: float, batch x 4 x 4
    - verts: float, batch x 3 x nverts
    """
    b, _, nverts = verts.shape
    # assert mvp.size(0) == b
    homo_verts = torch.cat([
        verts,
        torch.ones([b, 1, nverts], dtype=verts.dtype, device=verts.device)
    ], 1)  # b x 4 x nverts
    homo_verts = homo_verts.view(b, 4, -1)
    homo_trans_verts = torch.matmul(mvp, homo_verts)  # b x 4 x nverts
    trans_verts = homo_trans_verts[:, :3, :] / homo_trans_verts[:, [3], :]
    return trans_verts


def yaw_pitch_roll_to_quaternion(yaw, pitch, roll):
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)

    return torch.stack([
        cy * sr * cp - sy * cr * sp, cy * cr * sp + sy * sr * cp,
        sy * cr * cp - cy * sr * sp, cy * cr * cp + sy * sr * sp
    ], -1)


def rot_heads_to_quaternion(rot_heads):
    return yaw_pitch_roll_to_quaternion(rot_heads[:, 2], rot_heads[:, 0],
                                        rot_heads[:, 1])


@functools.lru_cache()
def _standard_face_pts():
    pts = np.array(
        [196.0, 226.0, 316.0, 226.0, 256.0, 286.0, 220.0, 360.4, 292.0, 360.4],
        np.float32) / 256.0 - 1.0
    return np.reshape(pts, (5, 2))


def compute_3d_face_align_matrices(mvps, face_landmark_vert_pos, imsize,
                                   target_face_scale):
    """
    inputs:
    - mvps: b x 4 x 4
    - face_landmark_vert_pos: b x 3 x 5

    outputs:
    - face_align_matrices: b x 4 x 4
    """
    std_pts = _standard_face_pts()  # [-1, 1]
    target_pts = std_pts * target_face_scale
    target_pts[:, 1] = -target_pts[:, 1]

    face_landmark_vert_pos_proj = transform_mesh_verts(
        mvps, face_landmark_vert_pos)  # [-1, 1]

    face_landmark_vert_pos_proj_np = face_landmark_vert_pos_proj.cpu().numpy()

    face_align_matrices_np = [None] * mvps.size(0)
    tform = transform.SimilarityTransform()
    for i in range(mvps.size(0)):
        fa_points = face_landmark_vert_pos_proj_np[i, :2, :].transpose([1, 0])
        tform.estimate(fa_points, target_pts)  # tform.params: 3 x 3
        m3 = tform.params
        # print(m3)
        face_align_matrices_np[i] = np.array(
            [[m3[0, 0], m3[0, 1], 0, m3[0, 2]],
             [m3[1, 0], m3[1, 1], 0, m3[1, 2]], [0, 0, 1, 0],
             [m3[2, 0], m3[2, 1], 0, m3[2, 2]]],
            dtype=np.float32)

    face_align_matrices = torch.from_numpy(
        np.stack(face_align_matrices_np, axis=0)).to(mvps.device)  # b x 4 x 4
    return face_align_matrices


def render(mvps,
           strands,
           body_mesh_vert_pos,
           body_mesh_face_inds,
           imsize,
           expansion,
           align_face=False,
           target_face_scale=1.0):
    """
    inputs:
    - mvps: b x 4 x 4
    - strands: b x 3 x sh x sw x sn
    - body_mesh_vert_pos: b x 3 x nverts_body
    - body_mesh_face_inds: b x 3 x nfaces_body
    - imsize: int
    - expansion: float

    outputs:
    - dxdy: b x 2 x imsize x imsize
    - strand_mask: b x imsize x imsize
    - face_mask: b x imsize x imsize
    - zs: b x imsize x imsize
    - strand_vis: b x sh x sw x sn
    - mvps: the aligned mvps
    - rd_strand_inds: b x 3(si,sj,pid) x h x w
    """

    assert mvps.size(1) == mvps.size(2) == 4
    assert strands.size(1) == 3
    assert body_mesh_vert_pos.size(1) == 3
    assert body_mesh_face_inds.size(1) == 3

    strand_num_points = haya_ext.count_strand_points(strands)

    if align_face:
        face_landmark_inds = [2463 - 5, 2466 - 5, 2455 - 5, 2503 - 5, 2506 - 5]
        face_landmarks = body_mesh_vert_pos[:, :, face_landmark_inds]
        face_align_matrices = compute_3d_face_align_matrices(
            mvps, face_landmarks, imsize, target_face_scale)
        mvps = torch.matmul(face_align_matrices, mvps)

    strands = transform_strands(mvps, strands)
    strands = strands * imsize / 2 + imsize / \
        2  # from [-1, +1] to image pixels

    body_mesh_vert_pos = transform_mesh_verts(mvps, body_mesh_vert_pos)
    body_mesh_vert_pos = body_mesh_vert_pos * imsize / 2 + imsize / 2

    zs, rd_strand_inds, rd_strand_lambdas, rd_face_inds, _ = \
        haya_ext.rasterize_strands(
            strands, strand_num_points, body_mesh_vert_pos, body_mesh_face_inds,
            imsize, imsize, expansion, 300, 1000, True, True)
    torch.cuda.synchronize()

    strand_mask = rd_strand_lambdas >= 0
    face_mask = rd_face_inds >= 0

    p1s, p2s = haya_ext.render_map_strands(strands, rd_strand_inds)
    torch.cuda.synchronize()

    _, _, sh, sw, sn = strands.shape
    strand_vis = haya_ext.count_visible_strands(rd_strand_inds, sh, sw, sn)

    ori_xyz = p2s - p1s  # b x 3 x imsize x imsize
    ori_xy = ori_xyz[:, :2, :, :]
    dxdy = ori_xy / \
        torch.max(torch.norm(ori_xy, None, dim=1, keepdim=True),
                  torch.tensor(1e-5, device=strands.device))

    return dxdy, strand_mask, face_mask, zs, strand_vis, mvps, rd_strand_inds


@functools.lru_cache()
def carnonical_face_landmark_positions():
    return torch.tensor(
        [[-0.062653, 1.76069, 0.092889], [0.058412, 1.76069, 0.092889],
         [-0.00212, 1.69329, 0.110603], [-0.039022, 1.65229, 0.091572],
         [0.034782, 1.65229, 0.091572]],
        dtype=torch.float32)


def compute_strand_lengths(strands):
    """ compute strand lengths
    Inputs:
    - strands: b x 3 x sh x sw x sn
    Returns:
    - lengths: b x sh x sw x sn
    """
    lens = (strands[:, :, :, :, :-1] - strands[:, :, :, :, 1:]).norm(
        None, dim=1, keepdim=False)
    return torch.cat([lens[:, :, :, [0]], lens], dim=-1)


def compute_single_collision_loss(strands, lens, x, y, z, a, b, c):
    """ single collision loss
    Inputs:
    - strands: b x 3 x sh x sw x sn
    Returns:
    - loss: b
    """
    hair_x = torch.pow(strands[:, 0] - x, 2) / \
        (a * a)  # b x sh x sw x sn
    hair_y = torch.pow(strands[:, 1] - y, 2) / (b * b)
    hair_z = torch.pow(strands[:, 2] - z, 2) / (c * c)
    hair_c = 1 - hair_x - hair_y - hair_z  # b x sh x sw x sn

    collision = lens * torch.max(hair_c, torch.zeros_like(hair_c))

    batch = collision.size(0)
    return collision.view(batch, -1).mean(-1)


def compute_collision_loss(strands, lens, gt_strand=None):
    """ collision loss
    Inputs:
    - strands: b x 3 x sh x sw x sn
    - lens: b x sh x sw x sn
    Returns:
    - loss: b
    """
    # if gt_strand is not None:
    #     per_strands = gt_strand.permute(0, 2, 3, 4, 1)  # b sh sw sn 3
    #     mask_strands = torch.where(
    #         (per_strands[..., 0] == 0) & (per_strands[..., 1] == 0) &
    #         (per_strands[..., 2] == 0), torch.zeros_like(per_strands[..., 0]),
    #         torch.ones_like(per_strands[..., 0]))

    collision0 = compute_single_collision_loss(strands,
                                               lens,
                                               x=-0.002,
                                               y=1.746,
                                               z=0,
                                               a=0.09,
                                               b=0.120,
                                               c=0.11)
    collision1 = compute_single_collision_loss(strands,
                                               lens,
                                               x=-0.002,
                                               y=1.589,
                                               z=-0.043,
                                               a=0.044,
                                               b=0.060,
                                               c=0.045)
    collision2 = compute_single_collision_loss(strands,
                                               lens,
                                               x=-0.091,
                                               y=1.480,
                                               z=-0.059,
                                               a=0.130,
                                               b=0.065,
                                               c=0.065)
    collision3 = compute_single_collision_loss(strands,
                                               lens,
                                               x=0.091,
                                               y=1.480,
                                               z=-0.059,
                                               a=0.130,
                                               b=0.065,
                                               c=0.065)

    return collision0 + collision1 + collision2 + collision3


def compute_reconstruction_loss(pred_strands,
                                gt_strands,
                                vismap,
                                visweight,
                                invisweight,
                                padded_zero=False):
    """ 
    Inputs:
    - pred_strands, gt_strands: b x 3 x sh x sw x sn
    - vismap: b x sh x sw x sn
    Returns:
    - loss: b
    """
    diff = (pred_strands - gt_strands).norm(None, dim=1,
                                            keepdim=False)  # b x sh x sw x sn
    weighted_diff = torch.where(vismap, visweight * diff, invisweight * diff)
    if padded_zero:
        # ignore loss where point position is zeros
        per_strands = gt_strands.permute(0, 2, 3, 4, 1)  # b sh sw sn 3
        mask_strands = torch.where(
            (per_strands[..., 0] == 0) & (per_strands[..., 1] == 0) &
            (per_strands[..., 2] == 0), torch.zeros_like(per_strands[..., 0]),
            torch.ones_like(per_strands[..., 0]))
        weighted_diff *= mask_strands
    return weighted_diff.view(diff.size(0), -1).mean(1)


def compute_masked_reconstruction_loss(pred_strands, gt_strands, mask=None):
    """ 
    Inputs:
    - pred_strands, gt_strands: b x 3 x sh x sw x sn
    - mask: b x sh x sw x sn
    Returns:
    - loss: b
    """
    diff = (pred_strands - gt_strands).norm(None, dim=1,
                                            keepdim=False)  # b x sh x sw x sn
    if mask is None:
        return diff.view(diff.size(0), -1).mean(1)
    else:
        visdiff = torch.where(mask, diff, torch.zeros_like(diff))
        return visdiff.view(visdiff.size(0), -1).sum(1) / \
            mask.type(diff.dtype).view(mask.size(0), -1).sum(1)


def compute_reprojection_loss(pred_info2d, gt_info2d):
    """
    Inputs:
    - pred_info2d, gt_info2d: b x (gx, gy, mhair, mface) x h x w
    Returns:
    - loss: b
    """
    pred_ori_xy = pred_info2d[:, :2, :, :]
    gt_ori_xy = gt_info2d[:, :2, :, :]

    gt_hair_mask = gt_info2d[:, 2, :, :] > 0

    abs_cos_dists = 1.0 - torch.abs(torch.sum(pred_ori_xy * gt_ori_xy, dim=1))
    abs_cos_dists = torch.where(gt_hair_mask, abs_cos_dists,
                                torch.zeros_like(abs_cos_dists))
    loss = abs_cos_dists.view(abs_cos_dists.size(0), -1).mean(1)
    return loss


def mask_pred_strands(pred_strands, strand_len=None):
    # in: B,3,32,32,300 out: B,32,32
    B, C, H, W, N = pred_strands.size()
    if strand_len is not None:
        strand_num_points = strand_len
    else:
        strand_num_points = haya_ext.count_strand_points(pred_strands)
    strand_num_points = strand_num_points.unsqueeze(1).expand(
        B, C, H, W).unsqueeze(-1).repeat(1, 1, 1, 1, N)
    index_num_points = torch.arange(N).expand_as(pred_strands).float().to(
        strand_num_points.device)
    masked_pred_strands = torch.where(strand_num_points >= index_num_points,
                                      pred_strands,
                                      torch.zeros_like(pred_strands))
    return masked_pred_strands


def compute_strand_len_loss(gt_len, pred_len):
    '''
    Inputs:
    - pred_len, gt_len: B, H, W
    Output:
    - loss: b
    '''
    loss = F.mse_loss(pred_len, gt_len,
                      reduction='none').view(gt_len.size(0), -1).mean(1)
    return loss


def compute_evenliness_loss(lens):
    """
    Inputs:
    - lens: b x sh x sw x sn
    Returns:
    - loss: b
    """
    mean_lens = lens.mean(-1, keepdim=True)  # b x sh x sw x 1
    rel_diffs = torch.abs(
        lens / torch.max(mean_lens, torch.tensor(1e-5, device=lens.device)) -
        1.0)  # b x sh x sw x sn
    return rel_diffs.view(rel_diffs.size(0), -1).mean(1)


def compute_smooth_and_curvature_loss(strands):
    """
    Inputs:
    - strands: b x 3 x sh x sw x sn
    """
    directions = strands[:, :, :, :, :-1] - strands[:, :, :, :, 1:]
    dirs1 = directions[:, :, :, :-1]
    dirs1 = dirs1 / torch.max(dirs1.norm(None, dim=1, keepdim=True),
                              torch.tensor(1e-5, device=strands.device))
    dirs2 = directions[:, :, :, 1:]
    dirs2 = dirs2 / torch.max(dirs2.norm(None, dim=1, keepdim=True),
                              torch.tensor(1e-5, device=strands.device))

    coses = 1.0 - torch.sum(dirs1 * dirs2, dim=1)
    smooth_loss = coses.view(coses.size(0), -1).mean(1)

    # b x 3 x sh x sw x (sn-2)
    crosses = torch.cross(dirs1, dirs2, dim=1)
    crosses1 = crosses[:, :, :, :-1]
    crosses2 = crosses[:, :, :, 1:]

    curvature_loss = 1.0 - torch.sum(dirs1 * dirs2, dim=1)
    curvature_loss = curvature_loss.view(curvature_loss.size(0), -1).mean(1)

    return smooth_loss, curvature_loss


@functools.lru_cache()
def compute_cosine_transform(num_points, k):
    """
    Inputs:
    - num_points:
    - k: num of bases
    Returns:
    - psi: num_points x k, float
    - psi_inv: k x num_points, float
    """
    t, l = torch.meshgrid(
        (torch.arange(1, num_points + 1, dtype=torch.float32) /
         float(num_points), torch.arange(k, dtype=torch.float32)))
    psi = math.sqrt(2) * torch.cos(l * math.pi * t)  # num_points x k
    psi[:, 0] = 1

    # compute (psi^T psi)^-1 psi^T
    psi_t = psi.t()
    psi_inv = torch.matmul(torch.inverse(torch.matmul(psi_t, psi)), psi_t)
    return psi, psi_inv


def strands_to_c(strands, psi_inv):
    """
    Inputs:
    - strands: b x 3 x sh x sw x sn
    - psi_inv: k x sn
    Outputs:
    - c: b x 3 x sh x sw x k
    """
    b, _, sh, sw, sn = strands.shape
    k, _ = psi_inv.shape
    strands = strands.permute(0, 2, 3, 4,
                              1).contiguous().view(-1, sn,
                                                   3)  # (b x sh x sw) x sn x 3
    c = torch.matmul(psi_inv, strands)  # (b x sh x sw) x k x 3
    return c.view(b, sh, sw, k, 3).permute(0, 4, 1, 2, 3)


def c_to_strands(c, psi):
    """
    Inputs:
    - c: b x 3 x sh x sw x k
    - psi: sn x k
    Outputs:
    - strands: b x 3 x sh x sw x sn
    """
    sn, k = psi.shape
    b, _, sh, sw, _ = c.shape
    c = c.permute(0, 2, 3, 4, 1).contiguous().view(-1, k,
                                                   3)  # (b x sh x sw) x k x 3
    strands = torch.matmul(psi, c)  # (b x sh x sw) x sn x 3
    return strands.view(b, sh, sw, sn, 3).permute(0, 4, 1, 2, 3)


def smooth_l1_loss(a, b, delta=1.0):
    diff = torch.abs(a - b)
    return torch.where(diff * delta < 1.0, 0.5 * diff * diff * delta,
                       diff - 0.5 / delta)


def compute_cosine_loss(pred_c, gt_c):
    """ compute_cos_loss
    Inputs:
    - pred_c, gt_c: b x 3 x sh x sw x k
    """
    return smooth_l1_loss(pred_c, gt_c).view(pred_c.size(0), -1).mean(1)


def gaussian_smooth(strands):
    """
    Inputs:
    - strands: b x 3 x sh x sw x sn
    """
    b, _, sh, sw, sn = strands.shape
    strands = strands.view(b, 1, -1, sn)
    kernel = torch.tensor([0.125, 0.125, 0.5, 0.125, 0.125],
                          dtype=strands.dtype,
                          device=strands.device)
    kernel = kernel.view(1, 1, 1, 5)
    strands = F.conv2d(F.pad(strands, (2, 2, 0, 0), mode='replicate'), kernel)
    strands = strands.view(b, 3, sh, sw, sn)
    return strands


@functools.lru_cache()
def _make_accumulate_kernel(n):
    """
    Returns:
    - kernel: n x n, np
    """
    return np.tril(np.ones([n, n], dtype=np.float32), k=0)


def accumulate(p0_ts):
    """ accumulate
    Inputs:
    - p0_ts: b x 3 x sh x sw x sn, the first is the init ps, 
        the else all tangential vectors
    Returns:
    - ps: b x 3 x sh x sw x sn
    """
    batch, _, sh, sw, sn = p0_ts.shape

    # (b*sh*sw) x sn x 3
    x = p0_ts.permute(0, 2, 3, 4, 1).view(batch * sh * sw, sn, 3)

    # accumulate positions
    # sn x sn x 1
    kernel = torch.from_numpy(
        _make_accumulate_kernel(sn)).to(device=p0_ts.device).view(sn, sn, 1)

    # (b*sh*sw) x sn x 3
    y = F.conv1d(x, kernel)
    # b x 3 x sh x sw x sn
    ps = y.view(batch, sh, sw, sn, 3).permute(0, 4, 1, 2, 3)
    return ps


@functools.lru_cache()
def _make_differentiate_kernel(n):
    """
    Returns:
    - kernel: n x n, np
    """
    acc_kernel = _make_accumulate_kernel(n)
    return np.linalg.inv(acc_kernel)


def differentiate(ps):
    """ differentiate
    Inputs:
    - ps: b x 3 x sh x sw x sn
    Returns:
    - p0_ts: b x 3 x sh x sw x sn, the first is the init ps, 
        the else all tangential vectors
    """
    batch, _, sh, sw, sn = ps.shape

    # (b*sh*sw) x sn x 3
    x = ps.permute(0, 2, 3, 4, 1).view(batch * sh * sw, sn, 3)

    kernel = torch.from_numpy(_make_differentiate_kernel(sn),
                              dtype=ps.dtype,
                              device=ps.device).view(sn, sn, 1)

    # (b*sh*sw) x sn x 3
    y = F.conv1d(x, kernel)  # b x n x 3

    # b x 3 x sh x sw x sn
    p0_ts = y.view(batch, sh, sw, sn, 3).permute(0, 4, 1, 2, 3)

    return p0_ts


@functools.lru_cache()
def _get_gabor_conv_kernel2(out_channels,
                            ksize=6,
                            sigma=4.0,
                            lambd=5,
                            gamma=0,
                            psi=0):
    """
    inputs:
    - ksize: kernel size
    - sigma: gaussian deviation
    - theta: kernel orientation
    - lambd: sinusoid wavelength
    - gamma: spatial aspect ratio
    outputs:
    - kernel: out_channels x ksize x ksize
    """
    # import viz
    gabor_kernels = [None] * out_channels
    for i in range(out_channels):
        gabor_kernels[i] = cv2.getGaborKernel(
            (ksize, ksize),
            sigma,
            i * math.pi / out_channels - math.pi / 2,
            lambd,
            gamma,
            psi=psi,
            ktype=cv2.CV_32F)
        # print(f'i={i}, angle={i*math.pi/out_channels}')
        # plt.imshow(gabor_kernels[i])
        # plt.show()
    # out_channels x ksize x ksize
    return np.stack(gabor_kernels, axis=0)


def gabor_conv2d(x, out_channels):
    """ gabor_conv2d
    Inputs:
    - x: b x in_channels x h x w
    Outputs:
    - gabor_activations: b x out_channels x h x w
    """
    conv_weight = torch.from_numpy(_get_gabor_conv_kernel2(out_channels)).type(
        x.dtype).to(x.device).unsqueeze(1).repeat(1, x.size(1), 1, 1)
    return F.conv2d(x, conv_weight, padding=(conv_weight.size(-1) - 1) // 2)


def get_orientation_xy(gabor_activations, act_thres=3e-5):
    """ get_orientation_xy
    Inputs:
    - gabor_activitions: b x out_channels x h x w
    Outputs:
    - dxdy: b x 2 x h x w
    - thetas: b x h x w
    - max_ori_inds: b x h x w
    """
    gabor_activations = gabor_activations.abs()
    valid_mask = gabor_activations.max(dim=1)[0] > act_thres

    out_channels = gabor_activations.size(1)
    _, max_ori_inds = torch.max(gabor_activations, dim=1)  # b x h x w
    max_ori_inds = torch.where(valid_mask, max_ori_inds,
                               -torch.ones_like(max_ori_inds))

    thetas = max_ori_inds.type(
        gabor_activations.dtype) * math.pi / out_channels  # 0~pi
    zeros_pl = torch.zeros_like(thetas)
    thetas = torch.where(valid_mask, thetas, zeros_pl)

    nori_x = torch.where(valid_mask, torch.cos(thetas), zeros_pl)
    nori_y = torch.where(valid_mask, torch.sin(thetas), zeros_pl)
    return torch.stack([nori_x, nori_y], dim=1), thetas, max_ori_inds


if __name__ == "__main__":
    import argparse
    import haya_data
    import viz
    import matplotlib.pyplot as plt
    parser = argparse.ArgumentParser()
    parser.add_argument('name', type=str)
    args = parser.parse_args()
    name = args.name
    if name == 'gabor_weights':
        out_channels = 32
        for i in range(out_channels):
            gabor_kernel = cv2.getGaborKernel((6, 6),
                                              4,
                                              i * math.pi / out_channels,
                                              5,
                                              0,
                                              psi=0,
                                              ktype=cv2.CV_32F)
            # gabor_kernel2 = cv2.getGaborKernel(
            #     (31, 31), 4, i*math.pi/4/out_channels+math.pi, 0.8, 0.8,
            #     psi=0, ktype=cv2.CV_32F)
            print(f'i={i}')
            plt.imshow(np.concatenate([gabor_kernel], axis=1))
            plt.show()
    elif name == 'orientation':
        image = cv2.imread('circle.png')[:, :, ::-1] / 255.0
        image = torch.from_numpy(image).unsqueeze(0).permute(0, 3, 1,
                                                             2).float().cuda()
        gact = gabor_conv2d(image, 180)
        # haya_data.viz.show_single(torch.cat([gact[0, 7], gact[0, 24]], dim=1))
        # for i in range(gact.size(1)):
        #     haya_data.viz.show_single(gact[0, i])
        dxy, thetas, inds = get_orientation_xy(gact, 3e-5)
        # plt.imshow((dxy[0].permute(1,2,0).cpu().numpy()[:,:,[0,1,1]] + 1) / 2)
        # plt.show()
        cv2.imwrite(
            'circle_viz.jpg',
            (dxy[0].permute(1, 2, 0).cpu().numpy()[:, :, [0, 1, 1]] + 1) / 2 *
            255)
