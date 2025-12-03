# webrm
This is an cut down version of our [Vulkan based viewer](https://github.com/half-potato/vkrm) to allow wider support. Training code [here](https://github.com/half-potato/radiance_meshes)

It `tet.js` can be used instead of `webgpu.js`, but it drastically reduces quality because I don't understand how GLSL precision works.

# Limitations
- No Mesh shading. Mesh shaders in webgpu do not support wave intrinsics yet, so there is very little benefit.
- WebGPU seems to be significantly slower? Viewer is 5x slower than the native Vulkan renderer.
