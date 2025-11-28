#version 300 es
precision highp float;
const float FLT_MAX = 3.402823466e+38;

uniform vec3 rayOrigin;

flat in float v_tetDensity;
flat in vec3 v_baseColor;
in float v_dc_dt;
in vec4 v_planeNumerators;
in vec4 v_planeDenominators;
in vec3 v_rayDir;
in vec3 vertex;
flat in vec3 v_tetAnchor; 
flat in vec3 v_grad;

out vec4 fragColor;

const uvec3 kTetTriangles[4] = uvec3[4](
    /*
     *    y
     *    ^
     *    |
     *    *v2    z
     *    |     /
     *    |    *v3
     *    |   /
     *    |  /
     *    | /          v1
     *    v0------------*--->x
     *
     */
    uvec3(0, 2, 1),
    uvec3(1, 2, 3),
    uvec3(0, 3, 2),
    uvec3(3, 0, 1)
);


float phi(float x) {
    if (abs(x) < 1e-6) {
        return 1.0f - x / 2.0f;
    }
    return (1.0f - exp(-x)) / x;
}

vec4 compute_integral(vec3 c0, vec3 c1, float ddt) {
    float alpha = exp(-ddt);
    float phi_val = phi(ddt);
    float w0 = phi_val - alpha;
    float w1 = 1.0f - phi_val;
    vec3 C = w0 * c0 + w1 * c1;
    return vec4(C.x, C.y, C.z, 1.f-alpha);
}

void main () {
    float d = length(v_rayDir);
    vec4 planeDenominators = v_planeDenominators / d;
    // float dc_dt = v_dc_dt / d;

    float opticalDepth = v_tetDensity;
    vec4 all_t = v_planeNumerators / planeDenominators;

    vec4 t_enter = mix(vec4(-FLT_MAX), all_t, greaterThan(planeDenominators, vec4(0.0)));
    vec4 t_exit  = mix(vec4(FLT_MAX), all_t, lessThan(planeDenominators, vec4(0.0)));

    vec2 t = vec2(
        max(t_enter.x, max(t_enter.y, max(t_enter.z, t_enter.w))),
        min(t_exit.x, min(t_exit.y, min(t_exit.z, t_exit.w)))
    );

    opticalDepth *= max(t.y - t.x, 0.f);

    vec3 N = v_rayDir / d;
    vec3 pos_enter = rayOrigin + N * t.x;
    vec3 local_diff = pos_enter - v_tetAnchor;
    float local_offset = dot(v_grad, local_diff);
    vec3 c_start = max(v_baseColor + local_offset, 0.0f);
    float dc_dt = dot(v_grad, N); // Change in color per unit of ray length
    vec3 c_end   = max(c_start + dc_dt * (t.y-t.x), 0.0f);

    // vec3 c_start = max(v_baseColor + dc_dt * t.x, 0.f);
    // vec3 c_end = max(v_baseColor + dc_dt * t.y, 0.f);
    fragColor = compute_integral(c_end, c_start, opticalDepth);
}
