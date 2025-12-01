import tinyplypy
import numpy as np
import argparse
from io import BytesIO
import gzip
import math

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]   


def RGB2SH(rgb):
    return (rgb - 0.5) / C0

def SH2RGB(sh):
    return sh * C0 + 0.5

def eval_sh(deg: int, sh, dirs):
    """
    Evaluate spherical harmonics at unit directions
    using hardcoded SH polynomials.
    Works with torch/np/jnp.
    ... Can be 0 or more batch dimensions.
    Args:
        deg: int SH deg. Currently, 0-3 supported
        sh: jnp.ndarray SH coeffs [..., C, (deg + 1) ** 2]
        dirs: jnp.ndarray unit directions [..., 3]
    Returns:
        [..., C]
    """
    assert deg <= 4 and deg >= 0
    coeff = (deg + 1) ** 2
    assert sh.shape[-1] >= coeff

    result = 0.28209479177387814 * sh[..., 0] + 0.5
    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result = (result -
                0.4886025119029199 * y * sh[..., 1] +
                0.4886025119029199 * z * sh[..., 2] -
                0.4886025119029199 * x * sh[..., 3])

        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (result +
                    1.0925484305920792 * xy * sh[..., 4] +
                    -1.0925484305920792 * yz * sh[..., 5] +
                    0.31539156525252005 * (2.0 * zz - xx - yy) * sh[..., 6] +
                    -1.0925484305920792 * xz * sh[..., 7] +
                    0.5462742152960396 * (xx - yy) * sh[..., 8])

            if deg > 2:
                result = (result +
                -0.5900435899266435 * y * (3 * xx - yy) * sh[..., 9] +
                2.890611442640554 * xy * z * sh[..., 10] +
                -0.4570457994644658 * y * (4 * zz - xx - yy)* sh[..., 11] +
                0.3731763325901154 * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12] +
                -0.4570457994644658 * x * (4 * zz - xx - yy) * sh[..., 13] +
                1.445305721320277 * z * (xx - yy) * sh[..., 14] +
                -0.5900435899266435 * x * (xx - 3 * yy) * sh[..., 15])

                if deg > 3:
                    result = (result + 2.5033429417967046 * xy * (xx - yy) * sh[..., 16] +
                            -1.7701307697799304 * yz * (3 * xx - yy) * sh[..., 17] +
                            0.9461746957575601 * xy * (7 * zz - 1) * sh[..., 18] +
                            -0.6690465435572892 * yz * (7 * zz - 3) * sh[..., 19] +
                            0.10578554691520431 * (zz * (35 * zz - 30) + 3) * sh[..., 20] +
                            -0.6690465435572892 * xz * (7 * zz - 3) * sh[..., 21] +
                            0.47308734787878004 * (xx - yy) * (7 * zz - 1) * sh[..., 22] +
                            -1.7701307697799304 * xz * (xx - 3 * yy) * sh[..., 23] +
                            0.6258357354491761 * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)) * sh[..., 24])
    return result


def load_sh(tetra_dict):
    sh_names = [k for k in tetra_dict.keys() if k.startswith("sh_")]
    N = tetra_dict[f"sh_0_r"].shape[0]
    sh_coeffs = np.zeros((N, len(sh_names) // 3, 3))
    for i in range(len(sh_names) // 3):
        sh_coeffs[:, i, 0] = tetra_dict[f"sh_{i}_r"]
        sh_coeffs[:, i, 1] = tetra_dict[f"sh_{i}_g"]
        sh_coeffs[:, i, 2] = tetra_dict[f"sh_{i}_b"]
    return sh_coeffs

def calculate_circumcenters(vertices):
    """
    Compute the circumcenter of a tetrahedron.
    
    Args:
        vertices: Tensor of shape (..., 4, 3) containing the vertices of the tetrahedron(a).
                 The first dimension can be batched.
    
    Returns:
        circumcenter: Tensor of shape (..., 3) containing the circumcenter coordinates
    """
    # Compute vectors from v0 to other vertices
    a = vertices[..., 1, :] - vertices[..., 0, :]  # v1 - v0
    b = vertices[..., 2, :] - vertices[..., 0, :]  # v2 - v0
    c = vertices[..., 3, :] - vertices[..., 0, :]  # v3 - v0
    
    # Compute squares of lengths
    aa = np.sum(a * a, axis=-1, keepdims=True)  # |a|^2
    bb = np.sum(b * b, axis=-1, keepdims=True)  # |b|^2
    cc = np.sum(c * c, axis=-1, keepdims=True)  # |c|^2
    
    # Compute cross products
    cross_bc = np.cross(b, c, axis=-1)
    cross_ca = np.cross(c, a, axis=-1)
    cross_ab = np.cross(a, b, axis=-1)
    
    # Compute denominator
    denominator = 2.0 * np.sum(a * cross_bc, axis=-1, keepdims=True)
    
    # Create mask for small denominators
    mask = np.abs(denominator) < 1e-12
    
    # Compute circumcenter relative to verts[0]
    relative_circumcenter = (
        aa * cross_bc +
        bb * cross_ca +
        cc * cross_ab
    ) / np.where(mask, np.ones_like(denominator), denominator)
    

    radius = np.linalg.norm(a - relative_circumcenter, axis=-1)
    
    # Return absolute position
    return vertices[..., 0, :] + relative_circumcenter, radius

def tet_volumes(tets):
    v0 = tets[:, 0]
    v1 = tets[:, 1]
    v2 = tets[:, 2]
    v3 = tets[:, 3]

    a = v1 - v0
    b = v2 - v0
    c = v3 - v0
    
    mat = np.stack((a, b, c), axis=1)
    det = np.linalg.det(mat)
    
    vol = det / 6.0
    return vol

def softplus(x, b=10):
    return 0.1*np.log(1+np.exp(10*x))

def process_ply_to_rmesh(ply_file_path):
    data = tinyplypy.read_ply(ply_file_path)
    vertices = np.stack([data['vertex']['x'], data['vertex']['y'], data['vertex']['z']], axis=1)
    indices = data['tetrahedron']['indices']
    s = data['tetrahedron']['s']
    mask = np.isnan(s)

    grd = np.stack([data['tetrahedron']['grd_x'], data['tetrahedron']['grd_y'], data['tetrahedron']['grd_z']], axis=1)
    sh_dat = load_sh(data['tetrahedron'])
    s = s[~mask]
    grd = grd[~mask]
    sh_dat = sh_dat[~mask]
    indices = indices[~mask]

    N = vertices.shape[0]
    M = indices.shape[0]

    tets = vertices[indices]
    cam1 = np.array([-4.0668, -0.5194,  0.9773]).reshape(1, 3)
    # cam1 = np.array([ 4.1367, -0.5304,  1.0696]).reshape(1, 3)
    cam1 = np.array([ 0.0, 3.0, -3.0]).reshape(1, 3)
    circumcenters, radius = calculate_circumcenters(tets.astype(np.double))
    circumcenters = circumcenters.astype(np.float32)
    centroid = tets.mean(axis=1)
    # dirs = -(vertices.mean(axis=0, keepdims=True) - centroid)
    dirs = -(cam1 - centroid)
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)

    density_t = s.astype(np.float16)

    # density_i = np.log(s)*20+100
    # density_i[density_i<=1] = 0
    # density_t = np.clip(density_i, 0, 255).astype(np.uint8)

    grd_t = grd.astype(np.float16)

    offset = np.sum(grd * (tets[:, 0] - centroid), axis=1, keepdims=True)

    # compress colors
    # colors = eval_sh(3, np.transpose(sh_dat, (0, 2, 1)), np.array([1, 0, 0]).reshape(1, 3))#dirs)
    sh_deg = math.sqrt(sh_dat.shape[1]) - 1
    colors = eval_sh(sh_deg, np.transpose(sh_dat, (0, 2, 1)), dirs)
    # sp_colors = 0.1*np.log(1+np.exp(10*colors)) + offset
    sp_colors = softplus(colors + offset)
    # plt.hist(sp_colors, bins=50)
    # plt.show()
    # rgb_t = ((sp_colors/4+0.5)*65535).clip(0, 65535).astype(np.uint16)
    # rgb_t = sp_colors.astype(np.float16)
    rgb_t = (sp_colors*255).clip(0, 255).astype(np.uint8)

    buffer = BytesIO()
    buffer.write(np.array([N, M]).astype(np.uint32).tobytes())
    buffer.write(vertices.tobytes())

    buffer.write(indices.tobytes())
    buffer.write(density_t.tobytes())
    buffer.write(rgb_t.tobytes())
    buffer.write(grd_t.tobytes())
    buffer.write(circumcenters.tobytes())

    compressed_bytes = gzip.compress(buffer.getvalue(), compresslevel=9)
    return compressed_bytes

    # return buffer.getvalue()


def save_rmesh_file(rmesh_data, output_path):
    with open(output_path, "wb") as f:
        f.write(rmesh_data)


def main():
    parser = argparse.ArgumentParser(description="Convert PLY files to SPLAT format.")
    parser.add_argument(
        "input_files", nargs="+", help="The input PLY files to process."
    )
    parser.add_argument(
        "--output", "-o", default="output.rmesh", help="The output RMESH file."
    )
    args = parser.parse_args()
    for input_file in args.input_files:
        print(f"Processing {input_file}...")
        rmesh_data = process_ply_to_rmesh(input_file)
        output_file = (
            args.output if len(args.input_files) == 1 else input_file + ".rmesh"
        )
        save_rmesh_file(rmesh_data, output_file)
        print(f"Saved {output_file}")


if __name__ == "__main__":
    main()
