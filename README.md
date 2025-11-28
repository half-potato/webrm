# webrm
This is an cut down version of our Vulkan based viewer to allow wider support. Ideally, this would use `wgpu` to allow for GPU based sorting and mesh shaders, but `wgpu` support is not very wide.

# Limitations
- No Mesh shading. This results in lower performance
- No spherical harmonics because of no compute shaders. Lower quality.
- CPU based sorting. This causes popping.
- GLSL imprecision. Floating point acts strangely across different platforms. 
- My inexperience with WebGL. This should really be considered more of a demo.

Inspired by antimatter15's 3DGS viewer.
