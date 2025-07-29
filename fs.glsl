#version 300 es
precision highp float;
const float FLT_MAX = 3.402823466e+38;

in vec3 v_rayDir;
flat in float v_tetDensity;
flat in vec3 v_baseColor;
in float v_dc_dt;
in vec4 v_planeNumerators;
in vec4 v_planeDenominators;

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

vec3 compute_integral(vec3 c0, vec3 c1, float d_dt) {
    float alpha = exp(-d_dt);
    float X = (-d_dt*alpha + 1.f - alpha);
    float Y = (d_dt-1.f) + alpha;
    return (X*c0+Y*c1) / d_dt;
}

float compute_integral_1D(float c0, float c1, float d_dt) {
    float alpha = exp(-d_dt);
    float X = (-d_dt*alpha + 1.f - alpha);
    float Y = (d_dt-1.f) + alpha;
    return (X*c0+Y*c1) / d_dt;
}

float integrate_channel(
    float t_n, float t_f,
    float c_at_t0, float dc_dt, float density)
{

    // Find where the linear color function C(t) would cross zero.
    float t_zero = clamp(-(c_at_t0 / dc_dt), t_n, t_f);

    // Determine the start and end of the segment where C(t) > 0.
    // This replaces the main "if (change_within)" and nested branches.
    float t_start = (dc_dt > 0.0f) ? t_zero : t_n;
    float t_end   = (dc_dt < 0.0f) ? t_zero : t_f;

    // Clamp this "positive" segment to the actual integration bounds [t_n, t_f].
    float dt_pos_segment = t_end - t_start;
    float d_dt = dt_pos_segment * density;
    if (d_dt < 1e-3f) {
        return 0.f;
    }

    // Calculate transmittance through the initial "zero-color" segment [t_n, t_start].
    float dt_zero_segment = t_start - t_n;
    float T_zero_segment = exp(-density * dt_zero_segment);

    // Calculate the integral over the clamped positive segment [t_start, t_end].
    float c_start = c_at_t0 + dc_dt * t_start;
    float c_end   = c_at_t0 + dc_dt * t_end;

    // The final result is the attenuated integral over the positive part.
    return T_zero_segment * compute_integral_1D(c_end, c_start, d_dt);
}

void main () {
    float d = length(v_rayDir);
    vec3 rayDir = v_rayDir / d;
    vec4 planeDenominators = v_planeDenominators / d;
    float dc_dt = v_dc_dt / d;

    float opticalDepth = v_tetDensity;
    vec4 all_t = v_planeNumerators / planeDenominators;

    vec4 t_enter = mix(vec4(-FLT_MAX), all_t, greaterThan(planeDenominators, vec4(0.0)));
    vec4 t_exit  = mix(vec4(FLT_MAX), all_t, lessThan(planeDenominators, vec4(0.0)));

    vec2 t = vec2(
        max(t_enter.x, max(t_enter.y, max(t_enter.z, t_enter.w))),
        min(t_exit.x, min(t_exit.y, min(t_exit.z, t_exit.w)))
    );

    opticalDepth *= t.y - t.x;
    float T = exp(-opticalDepth);
    vec3 c_start = v_baseColor + dc_dt * t.x;
    vec3 c_end   = v_baseColor + dc_dt * t.y;
    vec3 C = compute_integral(c_start, c_end, opticalDepth);
    fragColor = vec4(
        // integrate_channel(t.x, t.y, v_baseColor.r, dc_dt, v_tetDensity),
        // integrate_channel(t.x, t.y, v_baseColor.g, dc_dt, v_tetDensity),
        // integrate_channel(t.x, t.y, v_baseColor.b, dc_dt, v_tetDensity),
        C.r, C.g, C.b,
        // v_baseColor.r,
        // v_baseColor.g,
        // v_baseColor.b,
        1.f-T);
}
