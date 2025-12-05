# webrm
This is an cut down version of our [Vulkan based viewer](https://github.com/half-potato/vkrm) to allow wider support. Training code [here](https://github.com/half-potato/radiance_meshes)
Use `convert.py` to compress a `ply` file to a `rmesh` file. Uses [tinyplypy](https://github.com/half-potato/tinyplypy)

# Limitations
- No Mesh shading. Mesh shaders in webgpu do not support wave intrinsics yet, so there is very little benefit.
- WebGPU seems to be significantly slower? Viewer is 5x slower than the native Vulkan renderer.
