#version 300 es
precision highp float;
precision highp usampler2D;
precision highp sampler2D; // Added for the new gradientTexture type

uniform mat4 viewProjection;
uniform vec3 rayOrigin;

uniform sampler2D verticesTexture;
uniform ivec2 verticesTextureSize;

uniform usampler2D indicesTexture;
uniform ivec2 indicesTextureSize;

uniform sampler2D densityTexture;
uniform ivec2 densityTextureSize;

uniform sampler2D colorTexture;
uniform ivec2 colorTextureSize;

uniform sampler2D gradientTexture;
uniform ivec2 gradientTextureSize;

in uint tetId;
in uint vertexIdInTet;

out vec3 v_rayDir;
flat out float v_tetDensity;
flat out vec3 v_baseColor;
out float v_dc_dt;
out vec4 v_planeNumerators;
out vec4 v_planeDenominators;

const uvec3 kTetTriangles[4] = uvec3[4](
    /*
     * y
     * ^
     * |
     * *v2    z
     * |     /
     * |    *v3
     * |   /
     * |  /
     * | /         v1
     * v0------------*--->x
     *
     */
    uvec3(0, 2, 1),
    uvec3(1, 2, 3),
    uvec3(0, 3, 2),
    uvec3(3, 0, 1)
);

ivec2 getTexCoord(uint id, ivec2 size) {
    return ivec2(id % uint(size.x), id / uint(size.x));
}

vec3 fetchVertex(uint tet_id, uint vert_idx) {
    ivec2 indicesCoord = getTexCoord(tet_id, indicesTextureSize);
    uint v_idx = texelFetch(indicesTexture, indicesCoord, 0)[vert_idx];
    
    ivec2 vertexCoord = getTexCoord(v_idx, verticesTextureSize);
    return texelFetch(verticesTexture, vertexCoord, 0).xyz;
}


void main () {
    uint triId = vertexIdInTet / 3u;
    uint vertInTri = vertexIdInTet % 3u;
    uint localVertexIndex = kTetTriangles[triId][vertInTri];
    ivec2 indicesCoord = getTexCoord(tetId, indicesTextureSize);
    vec3 verts[4];

    for (int i=0; i<4; i++) {
        uint v_idx = texelFetch(indicesTexture, indicesCoord, 0)[i];
        ivec2 vertexCoord = getTexCoord(v_idx, verticesTextureSize);
        verts[i] = texelFetch(verticesTexture, vertexCoord, 0).xyz;
    }

    vec3 worldPos = verts[localVertexIndex];
    
    v_rayDir = worldPos - rayOrigin;

    ivec2 densityCoord = getTexCoord(tetId, densityTextureSize);
    // float compressed_density = float(texelFetch(densityTexture, densityCoord, 0).r);
    // v_tetDensity = exp((compressed_density - 100.0) / 20.0);
    v_tetDensity = texelFetch(densityTexture, densityCoord, 0).r;

    ivec2 gradCoord = getTexCoord(tetId, gradientTextureSize);
    vec3 grad = texelFetch(gradientTexture, gradCoord, 0).rgb;

    float offset2 = dot(rayOrigin - verts[0], grad);

    ivec2 colorCoord = getTexCoord(tetId, colorTextureSize);
    // v_baseColor = ((vec3(texelFetch(colorTexture, colorCoord, 0).rgb) / 65536.f) - 0.5) * 4.f + offset2;
    v_baseColor = texelFetch(colorTexture, colorCoord, 0).rgb + offset2;
    // v_baseColor = ((sp_colors/4+0.5)*65536).clip(0, 65536).astype(np.uint16)
    
    v_dc_dt = dot(grad, v_rayDir);
    // Calculate intersection information
    for (uint i = 0u; i < 4u; i++) {
        // outward facing normal
        vec3 n = cross(
            verts[kTetTriangles[i][2]] - verts[kTetTriangles[i][0]],
            verts[kTetTriangles[i][1]] - verts[kTetTriangles[i][0]]);
        v_planeNumerators[i] = dot(n, verts[kTetTriangles[i][0]] - rayOrigin);
        v_planeDenominators[i] = dot(n, v_rayDir);
    }

    gl_Position = viewProjection * vec4(worldPos, 1.0);
}

