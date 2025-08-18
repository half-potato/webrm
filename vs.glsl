#version 300 es
precision highp float;
precision highp usampler2D;
precision highp sampler2D;

// UNIFORMS (same as before)
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

// INPUT: Only the per-instance tetId is needed now.
in uint tetId;

// OUTPUTS (same as before)
flat out float v_tetDensity;
flat out vec3 v_baseColor;
out float v_dc_dt;
out vec4 v_planeNumerators;
out vec4 v_planeDenominators;
out vec3 v_rayDir;
out vec3 vertex;

ivec2 getTexCoord(uint id, ivec2 size) {
    return ivec2(id % uint(size.x), id / uint(size.x));
}

void main () {
    // 1. Fetch the 4 world-space vertex positions for the CURRENT INSTANCE using tetId
    vec3 verts[4];
    // ivec2 indicesCoord = getTexCoord(tetId, indicesTextureSize);

    ivec2 idxTC = getTexCoord(tetId, indicesTextureSize);
    uvec4 idx4  = texelFetch(indicesTexture, idxTC, 0);

    for (int i=0; i<4; i++) {
        uint v_idx = idx4[i];
        ivec2 vertexCoord = getTexCoord(v_idx, verticesTextureSize);
        verts[i] = texelFetch(verticesTexture, vertexCoord, 0).xyz;
    }

    vec3 worldPos = verts[gl_VertexID];
    vertex = worldPos;

    v_rayDir = normalize(worldPos - rayOrigin);

    ivec2 densityCoord = getTexCoord(tetId, densityTextureSize);
    v_tetDensity = texelFetch(densityTexture, densityCoord, 0).r;

    ivec2 gradCoord = getTexCoord(tetId, gradientTextureSize);
    vec3 grad = texelFetch(gradientTexture, gradCoord, 0).rgb;

    float offset2 = dot(rayOrigin - verts[0], grad);

    ivec2 colorCoord = getTexCoord(tetId, colorTextureSize);
    vec3 base_color_from_texture = texelFetch(colorTexture, colorCoord, 0).rgb;
    v_baseColor = base_color_from_texture + offset2;

    v_dc_dt = dot(grad, v_rayDir);

    // Re-create kTetTriangles logic locally for plane calculations
    const uvec3 kTetPlanes[4] = uvec3[4](
        uvec3(0, 2, 1), uvec3(1, 2, 3), uvec3(0, 3, 2), uvec3(3, 0, 1)
    );
    for (uint i = 0u; i < 4u; i++) {
        vec3 n = cross(
            verts[kTetPlanes[i][2]] - verts[kTetPlanes[i][0]],
            verts[kTetPlanes[i][1]] - verts[kTetPlanes[i][0]]);
        v_planeNumerators[i] = dot(n, verts[kTetPlanes[i][0]] - rayOrigin);
        v_planeDenominators[i] = dot(n, v_rayDir);
    }

    gl_Position = viewProjection * vec4(worldPos, 1.0);
}

