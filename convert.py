import tinyplypy
import numpy as np
import argparse
from io import BytesIO
import gzip
import math
from pathlib import Path
import json
import struct
from pyquaternion import Quaternion


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

def compress_matrix(data):
    # 1. Reshape to (Samples, Features) -> (8000000, 48)
    X = data.reshape(data.shape[0], -1)

    # 2. Center the data (crucial for PCA/SVD interpretation)
    mean_vec = np.mean(X, axis=0)
    X_centered = X - mean_vec

    # 3. Compute Covariance Matrix (48 x 48)
    # We compute C = (X.T @ X) / (N-1). This is fast and low memory.
    cov_matrix = np.dot(X_centered.T, X_centered) / (X_centered.shape[0] - 1)

    # 4. Eigen Decomposition on the small (48x48) matrix
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # 5. Sort eigenvectors by eigenvalues (descending)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sorted_indices]
    eigenvalues = eigenvalues[sorted_indices]
    print(eigenvalues)

    # 6. Compress: Keep top 'k' components (e.g., k=12 for 4x compression)
    k = 12
    top_vectors = eigenvectors[:, :k]

    # Project data onto principal components
    # Result shape: (8000000, 12)
    compressed_data = np.dot(X_centered, top_vectors)
    # To Reconstruct:
    reconstructed = np.dot(compressed_data, top_vectors.T) + mean_vec
    reconstructed = reconstructed.reshape(-1, 16, 3)
    print(np.abs(reconstructed - data).sum(axis=1).sum(axis=1).mean())
    return compressed_data, top_vectors.T, mean_vec



def process_ply_to_rmesh(ply_file_path, starting_cam, out_deg):
    data = tinyplypy.read_ply(ply_file_path)
    vertices = np.stack([data['vertex']['x'], data['vertex']['y'], data['vertex']['z']], axis=1)
    indices = data['tetrahedron']['indices']
    s = data['tetrahedron']['s']
    mask = np.isnan(s) | (s < 1e-3)

    grd = np.stack([data['tetrahedron']['grd_x'], data['tetrahedron']['grd_y'], data['tetrahedron']['grd_z']], axis=1)
    sh_dat = load_sh(data['tetrahedron']) # Shape: (N, 16, 3)
    sh_dat = sh_dat[:, :(out_deg+1)**2]
    sh_comp = compress_matrix(sh_dat)

    # Filter masks
    s = s[~mask]
    grd = grd[~mask]
    sh_dat = sh_dat[~mask]
    indices = indices[~mask]

    N = vertices.shape[0]
    M = indices.shape[0]

    tets = vertices[indices]
    
    # 1. Prepare Density (Log space -> uint8)
    # density_i = np.log(s.clip(min=1e-3))*20+100
    # density_i[density_i<=1] = 0
    # density_t = np.clip(density_i, 0, 255).astype(np.uint8)
    density_t = s.astype(np.float16)

    # 2. Prepare Gradients
    grd_t = grd.astype(np.float16)

    # 3. Prepare SH as Half Float (float16)
    # Cast directly to float16, no normalization to 0-255 required
    sh_half = sh_dat.astype(np.float16)
    
    # Transpose to Structure of Arrays: [Channel, Coeff, Tet]
    sh_soa = np.transpose(sh_half, (2, 1, 0))

    # Flatten: [R_C0_all... R_C15_all... G_C0_all...]
    sh_flat = sh_soa.flatten()

    buffer = BytesIO()
    buffer.write(np.array([N, M, out_deg, 0]).astype(np.uint32).tobytes())
    buffer.write(starting_cam.astype(np.float32).reshape(8).tobytes())
    buffer.write(vertices.tobytes())
    buffer.write(indices.tobytes())
    buffer.write(density_t.tobytes())
    current_pos = buffer.tell()
    pad_len = (4 - (current_pos % 4)) % 4
    buffer.write(b'\x00' * pad_len)

    # Write the float16 buffer
    buffer.write(sh_flat.tobytes())
    current_pos = buffer.tell()
    pad_len = (4 - (current_pos % 4)) % 4
    buffer.write(b'\x00' * pad_len)

    buffer.write(grd_t.tobytes())

    compressed_bytes = gzip.compress(buffer.getvalue(), compresslevel=9)
    return compressed_bytes


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def read_null_terminated_string(fid):
    """
    Reads characters from a file object until a null terminator is found.
    """
    char_list = []
    while True:
        char = fid.read(1)
        if char == b'' or char == b'\x00': # Stop on null terminator or end of file
            break
        char_list.append(char)
    return b''.join(char_list).decode("utf-8")

def read_extrinsics_binary(path_to_model_file, transform_matrix):
    """
    Reads the images.bin file, converts to GL coordinates, applies transform,
    and exports as [x, y, z, w] for JS compatibility.
    """
    images = []

    # Orthogonalize the transform matrix to prevent skewing
    u, _, vt = np.linalg.svd(transform_matrix[:3, :3])
    transform_matrix[:3, :3] = u @ vt

    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]

            # COLMAP qvec is [w, x, y, z]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]

            image_name = read_null_terminated_string(fid)

            num_points2D = read_next_bytes(fid, num_bytes=8,
                                           format_char_sequence="Q")[0]
            fid.seek(24 * num_points2D, 1) # Skip 2D points

            # 1. W2C (World-to-Camera) from COLMAP
            w2c = np.eye(4)
            w2c[:3, :3] = Quaternion(qvec).rotation_matrix
            w2c[:3, 3] = tvec

            # 2. C2W (Camera-to-World)
            c2w = np.linalg.inv(w2c)

            # 3. Coordinate System Conversion: COLMAP (Y-Down, Z-Fwd) -> GL (Y-Up, Z-Back)
            # Flip local Y and Z axes by multiplying the rotation columns
            c2w[:3, 1:3] *= -1

            # 4. Apply scene transformation
            new_c2w = transform_matrix @ c2w

            new_tvec = new_c2w[:3, 3]
            new_q = Quaternion(matrix=new_c2w[:3, :3])

            # 5. Reorder to [x, y, z, w] for JS consumption
            qvec_js = np.array([new_q.x, new_q.y, new_q.z, new_q.w])

            images.append(np.concatenate([new_tvec, qvec_js, np.array([0.0])], axis=0))

    return images

def main():
    parser = argparse.ArgumentParser(description="Convert PLY files to SPLAT format.")
    parser.add_argument(
        "input_files", nargs="+", help="The input PLY files to process."
    )
    parser.add_argument(
        "--output", "-o", default="output.rmesh", help="The output RMESH file."
    )
    parser.add_argument(
        "--degree", "-d", type=int, default=2, help="The degree of the final mesh"
    )
    args = parser.parse_args()
    for input_file in args.input_files:
        print(f"Processing {input_file}...")
        transform = np.loadtxt((Path(input_file).parent / "transform.txt"))
        with (Path(input_file).parent / "config.json").open('r') as f:
            js = json.load(f)
            path = Path(js['dataset_path']) / "sparse/0/images.bin"
            extrinsics = read_extrinsics_binary(str(path), transform)
        rmesh_data = process_ply_to_rmesh(input_file, extrinsics[0], args.degree)
        output_file = (
            args.output if len(args.input_files) == 1 else input_file + ".rmesh"
        )
        with open(output_file, "wb") as f:
            f.write(rmesh_data)
        print(f"Saved {output_file}")


if __name__ == "__main__":
    main()
