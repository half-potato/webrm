# webrm
This is an cut down version of our Vulkan based viewer to allow wider support. It `tet.js` can be used instead of `webgpu.js`, but it drastically reduces quality because I don't understand how GLSL precision works.

# Limitations
- No Mesh shading. Mesh shaders in webgpu do not support wave intrinsics yet, so there is very little benefit.
- No spherical harmonics. Yet. I think webgpu supports this.
- CPU based sorting. Yet. This causes popping during fast moves.