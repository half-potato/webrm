const computeSource = `
struct Uniforms {
    viewProjection: mat4x4<f32>,
    rayOrigin: vec3<f32>,      // align 16
    camPos: vec3<f32>,         // align 16
    screenWidth: f32,
    screenHeight: f32,
    tetCount: u32,             // align 16
    shDegree: u32,
};

struct DrawIndirectArgs {
    vertexCount: atomic<u32>,
    instanceCount: atomic<u32>, // We will atomicAdd this
    firstVertex: u32,
    firstInstance: u32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> vertices: array<f32>;
@group(0) @binding(2) var<storage, read> indices: array<u32>;
@group(0) @binding(3) var<storage, read_write> colors_compute: array<f32>;
@group(0) @binding(4) var<storage, read> gradients: array<f32>;
@group(0) @binding(5) var<storage, read> shCoeffs: array<u32>;
// New Bindings for Sorting/Culling
@group(0) @binding(6) var<storage, read> circumcenters: array<f32>;
@group(0) @binding(7) var<storage, read> circumradii: array<f32>;
@group(0) @binding(8) var<storage, read_write> sortKeys: array<u32>;   // Output for Radix Sort
@group(0) @binding(9) var<storage, read_write> sortValues: array<u32>; // Output for Radix Sort
@group(0) @binding(10) var<storage, read_write> indirectArgs: DrawIndirectArgs;

// --- SH Constants ---
const C0 = 0.28209479177387814;
const C1 = 0.4886025119029199;
const C2_0 = 1.0925484305920792;
const C2_1 = -1.0925484305920792;
const C2_2 = 0.31539156525252005;
const C2_3 = -1.0925484305920792;
const C2_4 = 0.5462742152960396;

fn getSHByte(channel: u32, coeff: u32, tetIdx: u32) -> f32 {
    let numCoeffs = (uniforms.shDegree + 1u) * (uniforms.shDegree + 1u);
    let channelOffset = channel * numCoeffs * uniforms.tetCount;
    let coeffOffset = coeff * uniforms.tetCount;
    let totalIndex = channelOffset + coeffOffset + tetIdx;
    let wordIndex = totalIndex / 4u;
    let byteIndex = totalIndex % 4u;
    let packedVal = shCoeffs[wordIndex];
    let shift = byteIndex * 8u;
    let u8Val = (packedVal >> shift) & 0xFFu;
    return ((f32(u8Val) / 127.5) - 1.0) * 2.0;
}

fn projectToNdc(pos: vec3<f32>, vp: mat4x4<f32>) -> vec4<f32> {
    let clip = vp * vec4<f32>(pos, 1.0);
    let invW = 1.0 / (clip.w + 0.000001);
    return vec4<f32>(clip.xyz * invW, clip.w);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    let tetId = global_id.x + (global_id.y * num_workgroups.x * 64u);
    if (tetId >= uniforms.tetCount) { return; }

    // --- 1. GEOMETRY PREP ---
    let i0 = indices[tetId * 4u + 0u] * 3u;
    let i1 = indices[tetId * 4u + 1u] * 3u;
    let i2 = indices[tetId * 4u + 2u] * 3u;
    let i3 = indices[tetId * 4u + 3u] * 3u;

    let v0 = vec3<f32>(vertices[i0], vertices[i0+1u], vertices[i0+2u]);
    let v1 = vec3<f32>(vertices[i1], vertices[i1+1u], vertices[i1+2u]);
    let v2 = vec3<f32>(vertices[i2], vertices[i2+1u], vertices[i2+2u]);
    let v3 = vec3<f32>(vertices[i3], vertices[i3+1u], vertices[i3+2u]);

    // --- 2. CULLING ---
    // Project all 4 vertices to NDC
    let p0 = projectToNdc(v0, uniforms.viewProjection);
    let p1 = projectToNdc(v1, uniforms.viewProjection);
    let p2 = projectToNdc(v2, uniforms.viewProjection);
    let p3 = projectToNdc(v3, uniforms.viewProjection);

    // Calculate AABB in NDC
    let minX = min(min(p0.x, p1.x), min(p2.x, p3.x));
    let maxX = max(max(p0.x, p1.x), max(p2.x, p3.x));
    let minY = min(min(p0.y, p1.y), min(p2.y, p3.y));
    let maxY = max(max(p0.y, p1.y), max(p2.y, p3.y));
    let minZ = min(min(p0.z, p1.z), min(p2.z, p3.z)); // WebGPU NDC Z is 0..1

    // Frustum Check
    var isVisible = true;
    if (maxX < -1.0 || minX > 1.0 || maxY < -1.0 || minY > 1.0 || minZ > 1.0) {
        isVisible = false;
    }

    // Screen Size Check
    let screenScaleX = uniforms.screenWidth;
    let screenScaleY = uniforms.screenHeight;
    let extX = (maxX - minX) * screenScaleX;
    let extY = (maxY - minY) * screenScaleY;
    if ((extX * extY) < 1.0) {
        isVisible = false;
    }

    // --- 3. KEY GENERATION ---
    if (!isVisible) {
        // Invisible: Set key to Max Uint so it sorts to the very end
        sortKeys[tetId] = 0xFFFFFFFFu;
        // We still write the ID, though it won't be drawn due to indirect count
        sortValues[tetId] = tetId; 
    } else {
        // Visible: Calculate Depth
        let cx = circumcenters[tetId * 3u];
        let cy = circumcenters[tetId * 3u + 1u];
        let cz = circumcenters[tetId * 3u + 2u];
        let r2 = circumradii[tetId]; // Radius Squared

        let diff = vec3<f32>(cx, cy, cz) - uniforms.camPos;
        // Depth = DistanceSq - RadiusSq (Distance to the closest point of sphere)
        // We want Painter's Algo: Draw Far to Near.
        // Radix Sort is Ascending (Small -> Large).
        // We want Large Depth first. 
        // Key = MaxUint - DepthBits.
        
        let distSq = dot(diff, diff);
        let depth = distSq - r2;
        
        // Convert float depth to uint preserving order
        // Since depth is likely positive (unless camera inside sphere), simple bitcast works.
        // If depth can be negative, we need to flip sign bit logic (omitted for brevity, assuming external view)
        let depthBits = bitcast<u32>(depth);
        
        // Invert bits to reverse sort order (Far -> Near)
        sortKeys[tetId] = ~depthBits; 
        sortValues[tetId] = tetId;

        // Atomically increment the instance count for the Draw call
        atomicAdd(&indirectArgs.instanceCount, 1u);

        // --- 4. SH EVALUATION (Only if visible? Or always? Let's do always for simplicity) ---
        // Optimally, you would put this in a separate pass running only on 'indirectArgs.instanceCount' items,
        // but doing it here prevents pipeline complexity.

        let centroid = (v0 + v1 + v2 + v3) * 0.25;
        let dir = normalize(centroid - uniforms.camPos); 
        let x = dir.x; let y = dir.y; let z = dir.z;

        var resultColor = vec3<f32>(0.0);

        for (var c = 0u; c < 3u; c++) {
            // Degree 0
            let sh0 = getSHByte(c, 0u, tetId);
            var val = C0 * sh0;
            // Degree 1
            let sh1 = getSHByte(c, 1u, tetId);
            let sh2 = getSHByte(c, 2u, tetId);
            let sh3 = getSHByte(c, 3u, tetId);
            val -= C1 * y * sh1;
            val += C1 * z * sh2;
            val -= C1 * x * sh3;
            // Degree 2
            let xx = x*x; let yy = y*y; let zz = z*z;
            let xy = x*y; let yz = y*z; let xz = x*z;
            let sh4 = getSHByte(c, 4u, tetId);
            let sh5 = getSHByte(c, 5u, tetId);
            let sh6 = getSHByte(c, 6u, tetId);
            let sh7 = getSHByte(c, 7u, tetId);
            let sh8 = getSHByte(c, 8u, tetId);
            val += C2_0 * xy * sh4;
            val += C2_1 * yz * sh5;
            val += C2_2 * (2.0 * zz - xx - yy) * sh6;
            val += C2_3 * xz * sh7;
            val += C2_4 * (xx - yy) * sh8;

            resultColor[c] = val + 0.5;
        }

        let gx = gradients[tetId * 3u + 0u];
        let gy = gradients[tetId * 3u + 1u];
        let gz = gradients[tetId * 3u + 2u];
        let grad = vec3<f32>(gx, gy, gz);
        let offset = dot(grad, v0 - centroid);
        let inputVal = resultColor + vec3<f32>(offset);
        let sp = 0.1 * log(1.0 + exp(10.0 * inputVal));

        colors_compute[tetId * 3u + 0u] = sp.x;
        colors_compute[tetId * 3u + 1u] = sp.y;
        colors_compute[tetId * 3u + 2u] = sp.z;

    }
}
`;

const wgslSource = `
struct Uniforms {
    viewProjection: mat4x4<f32>, // offset 0
    rayOrigin: vec3<f32>,        // offset 64 (align 16)
    camPos: vec3<f32>,           // offset 80 (align 16)
    tetCount: u32,               // offset 96 (align 16)
    // Implicit padding makes this struct 112 bytes total
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> vertices: array<f32>;
@group(0) @binding(2) var<storage, read> indices: array<u32>;
@group(0) @binding(3) var<storage, read> densities: array<f32>;
@group(0) @binding(4) var<storage, read> colors_vertex: array<f32>;
@group(0) @binding(5) var<storage, read> gradients: array<f32>;
@group(0) @binding(6) var<storage, read> sortedIndices: array<u32>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) tetDensity: f32,
    @location(1) @interpolate(flat) baseColor: vec3<f32>,
    @location(2) planeNumerators: vec4<f32>,
    @location(3) planeDenominators: vec4<f32>,
    @location(4) rayDir: vec3<f32>,
    @location(5) dc_dt: f32,
};

// Hardcoded indices for the 4 faces of a tetrahedron (12 verts total)
// Face 1: 0,2,1 | Face 2: 1,2,3 | Face 3: 0,3,2 | Face 4: 3,0,1
const kTetFaceIndices = array<u32, 12>(
    0u, 2u, 1u,
    1u, 2u, 3u,
    0u, 3u, 2u,
    3u, 0u, 1u
);

fn getVertex(index: u32) -> vec3<f32> {
    let i = index * 3u;
    return vec3<f32>(vertices[i], vertices[i+1u], vertices[i+2u]);
}

fn getColor(tetId: u32) -> vec3<f32> {
    let i = tetId * 3u;
    // Use the READ-ONLY buffer here
    return vec3<f32>(colors_vertex[i], colors_vertex[i+1u], colors_vertex[i+2u]);
}

fn getGradient(tetId: u32) -> vec3<f32> {
    let i = tetId * 3u;
    return vec3<f32>(gradients[i], gradients[i+1u], gradients[i+2u]);
}

@vertex
fn vs_main(@builtin(instance_index) instanceIdx: u32, @builtin(vertex_index) vertIdx: u32) -> VertexOutput {
    var out: VertexOutput;

    // 1. Get the actual Tet ID from the sorted list
    let tetId = sortedIndices[instanceIdx];

    // 2. Fetch the 4 indices for this tetrahedron
    // indices buffer is packed u32: [t0_v0, t0_v1, t0_v2, t0_v3, t1_v0...]
    let i0 = indices[tetId * 4u + 0u];
    let i1 = indices[tetId * 4u + 1u];
    let i2 = indices[tetId * 4u + 2u];
    let i3 = indices[tetId * 4u + 3u];

    var verts: array<vec3<f32>, 4>;
    verts[0] = getVertex(i0);
    verts[1] = getVertex(i1);
    verts[2] = getVertex(i2);
    verts[3] = getVertex(i3);

    // 3. Determine which vertex of the 12 we are drawing
    // kTetFaceIndices maps 0..11 to 0..3 local index
    let localIndex = kTetFaceIndices[vertIdx];
    let worldPos = verts[localIndex];

    out.rayDir = normalize(worldPos - uniforms.rayOrigin);
    out.tetDensity = densities[tetId];
    //out.grad = getGradient(tetId);
    //out.tetAnchor = verts[0];
    let grad = getGradient(tetId);
    out.dc_dt = dot(grad, out.rayDir);
    let offset2 = dot(grad, uniforms.rayOrigin - verts[0]);
    out.baseColor = getColor(tetId) + offset2;

    // 4. Compute Planes (Ray-Tet Intersection Math)
    // We recreate the geometry locally to compute normals
    // Planes: (0,2,1), (1,2,3), (0,3,2), (3,0,1)
    let p0 = vec3<u32>(0u, 2u, 1u);
    let p1 = vec3<u32>(1u, 2u, 3u);
    let p2 = vec3<u32>(0u, 3u, 2u);
    let p3 = vec3<u32>(3u, 0u, 1u);

    var planes = array<vec3<u32>, 4>(p0, p1, p2, p3);

    for (var i = 0u; i < 4u; i++) {
        let idxs = planes[i];
        let vA = verts[idxs.x];
        let vB = verts[idxs.y];
        let vC = verts[idxs.z];

        // Face normal
        let n = cross(vC - vA, vB - vA);

        out.planeNumerators[i] = dot(n, vA - uniforms.rayOrigin);
        out.planeDenominators[i] = dot(n, out.rayDir);
    }

    out.position = uniforms.viewProjection * vec4<f32>(worldPos, 1.0);
    return out;
}


// --- Fragment Shader Utils ---

fn phi(x: f32) -> f32 {
    if (abs(x) < 1e-6) {
        return 1.0 - x * 0.5;
    }
    return (1.0 - exp(-x)) / x;
}

fn compute_integral(c0: vec3<f32>, c1: vec3<f32>, ddt: f32) -> vec4<f32> {
    let alpha = exp(-ddt);
    let phi_val = phi(ddt);
    let w0 = phi_val - alpha;
    let w1 = 1.0 - phi_val;
    let C = w0 * c0 + w1 * c1;
    return vec4<f32>(C, 1.0 - alpha);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let d = length(in.rayDir);
    let planeDenom = in.planeDenominators / d;
    let dc_dt = in.dc_dt / d;

    var opticalDepth = in.tetDensity;
    let all_t = in.planeNumerators / planeDenom;

    // WebGPU doesn't have vector greaterThan/lessThan logic exactly like GLSL
    // We use select for component-wise checks.

    // t_enter: max of intersections where denom > 0
    var t_enter = vec4<f32>(-3.402823e38); // -FLT_MAX
    if (planeDenom.x > 0.0) { t_enter.x = all_t.x; }
    if (planeDenom.y > 0.0) { t_enter.y = all_t.y; }
    if (planeDenom.z > 0.0) { t_enter.z = all_t.z; }
    if (planeDenom.w > 0.0) { t_enter.w = all_t.w; }

    // t_exit: min of intersections where denom < 0
    var t_exit = vec4<f32>(3.402823e38); // FLT_MAX
    if (planeDenom.x < 0.0) { t_exit.x = all_t.x; }
    if (planeDenom.y < 0.0) { t_exit.y = all_t.y; }
    if (planeDenom.z < 0.0) { t_exit.z = all_t.z; }
    if (planeDenom.w < 0.0) { t_exit.w = all_t.w; }

    let t_min = max(t_enter.x, max(t_enter.y, max(t_enter.z, t_enter.w)));
    let t_max = min(t_exit.x, min(t_exit.y, min(t_exit.z, t_exit.w)));

    let dist = max(t_max - t_min, 0.0);
    opticalDepth *= dist;

    let c_start = max(in.baseColor + dc_dt * t_min, vec3<f32>(0.0));
    let c_end = max(in.baseColor + dc_dt * t_max, vec3<f32>(0.0));

    return compute_integral(c_end, c_start, opticalDepth);
}
`;

//
//
// --- Matrix Math Utilities ---
function multiply4(a, b) {
    let out = new Float32Array(16);
    let a00 = a[0], a01 = a[1], a02 = a[2], a03 = a[3], a10 = a[4], a11 = a[5], a12 = a[6], a13 = a[7], a20 = a[8], a21 = a[9], a22 = a[10], a23 = a[11], a30 = a[12], a31 = a[13], a32 = a[14], a33 = a[15];
    let b0 = b[0], b1 = b[1], b2 = b[2], b3 = b[3];
    out[0] = b0 * a00 + b1 * a10 + b2 * a20 + b3 * a30;
    out[1] = b0 * a01 + b1 * a11 + b2 * a21 + b3 * a31;
    out[2] = b0 * a02 + b1 * a12 + b2 * a22 + b3 * a32;
    out[3] = b0 * a03 + b1 * a13 + b2 * a23 + b3 * a33;
    b0 = b[4]; b1 = b[5]; b2 = b[6]; b3 = b[7];
    out[4] = b0 * a00 + b1 * a10 + b2 * a20 + b3 * a30;
    out[5] = b0 * a01 + b1 * a11 + b2 * a21 + b3 * a31;
    out[6] = b0 * a02 + b1 * a12 + b2 * a22 + b3 * a32;
    out[7] = b0 * a03 + b1 * a13 + b2 * a23 + b3 * a33;
    b0 = b[8]; b1 = b[9]; b2 = b[10]; b3 = b[11];
    out[8] = b0 * a00 + b1 * a10 + b2 * a20 + b3 * a30;
    out[9] = b0 * a01 + b1 * a11 + b2 * a21 + b3 * a31;
    out[10] = b0 * a02 + b1 * a12 + b2 * a22 + b3 * a32;
    out[11] = b0 * a03 + b1 * a13 + b2 * a23 + b3 * a33;
    b0 = b[12]; b1 = b[13]; b2 = b[14]; b3 = b[15];
    out[12] = b0 * a00 + b1 * a10 + b2 * a20 + b3 * a30;
    out[13] = b0 * a01 + b1 * a11 + b2 * a21 + b3 * a31;
    out[14] = b0 * a02 + b1 * a12 + b2 * a22 + b3 * a32;
    out[15] = b0 * a03 + b1 * a13 + b2 * a23 + b3 * a33;
    return out;
}

function invert4(a) {
    let out = new Float32Array(16);
    let a00 = a[0], a01 = a[1], a02 = a[2], a03 = a[3], a10 = a[4], a11 = a[5], a12 = a[6], a13 = a[7], a20 = a[8], a21 = a[9], a22 = a[10], a23 = a[11], a30 = a[12], a31 = a[13], a32 = a[14], a33 = a[15];
    let b00 = a00 * a11 - a01 * a10, b01 = a00 * a12 - a02 * a10, b02 = a00 * a13 - a03 * a10, b03 = a01 * a12 - a02 * a11, b04 = a01 * a13 - a03 * a11, b05 = a02 * a13 - a03 * a12, b06 = a20 * a31 - a21 * a30, b07 = a20 * a32 - a22 * a30, b08 = a20 * a33 - a23 * a30, b09 = a21 * a32 - a22 * a31, b10 = a21 * a33 - a23 * a31, b11 = a22 * a33 - a23 * a32;
    let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
    if (!det) return null;
    det = 1.0 / det;
    out[0] = (a11 * b11 - a12 * b10 + a13 * b09) * det;
    out[1] = (a02 * b10 - a01 * b11 - a03 * b09) * det;
    out[2] = (a31 * b05 - a32 * b04 + a33 * b03) * det;
    out[3] = (a22 * b04 - a21 * b05 - a23 * b03) * det;
    out[4] = (a12 * b08 - a10 * b11 - a13 * b07) * det;
    out[5] = (a00 * b11 - a02 * b08 + a03 * b07) * det;
    out[6] = (a32 * b02 - a30 * b05 - a33 * b01) * det;
    out[7] = (a20 * b05 - a22 * b02 + a23 * b01) * det;
    out[8] = (a10 * b10 - a11 * b08 + a13 * b06) * det;
    out[9] = (a01 * b08 - a00 * b10 - a03 * b06) * det;
    out[10] = (a30 * b04 - a31 * b02 + a33 * b00) * det;
    out[11] = (a21 * b02 - a20 * b04 - a23 * b00) * det;
    out[12] = (a11 * b07 - a10 * b09 - a12 * b06) * det;
    out[13] = (a00 * b09 - a01 * b07 + a02 * b06) * det;
    out[14] = (a31 * b01 - a30 * b03 - a32 * b00) * det;
    out[15] = (a20 * b03 - a21 * b01 + a22 * b00) * det;
    return out;
}

function decodeHalf(float16bits) {
    const exponent = (float16bits & 0x7C00) >> 10;
    const fraction = float16bits & 0x03FF;
    const sign = (float16bits & 0x8000) ? -1 : 1;

    if (exponent === 0) {
        return sign * Math.pow(2, -14) * (fraction / 1024);
    } else if (exponent === 0x1F) {
        return fraction ? NaN : sign * Infinity;
    }

    return sign * Math.pow(2, exponent - 15) * (1 + (fraction / 1024));
}


class Camera {
    constructor(position = [0, 5, -2], canvas) {
        this.canvas = canvas;

        // General properties
        this.fovY = 50; // Vertical field of view in degrees
        this.nearZ = 0.01;
        this.farZ = 1000.0;

        // Mode state
        this.mode = 'orbit'; // 'free' or 'orbit'
        this.upAxis = new Float32Array([0, 0, -1]); // Z is up

        // --- State for Free-Cam (Quaternion-based) ---
        this.position = new Float32Array(position);
        // Initial rotation quaternion is calculated to match the starting orbit view
        this.rotation = new Float32Array([0, 0, 0, 1]); 
        this.baseMoveSpeed = 0.5;
        this.rollSpeed = 1.5;

        // --- State for Orbit-Cam (Euler-based) ---
        this.orbitTarget = new Float32Array([0, 0, 0]);
        this.orbitDistance = Math.hypot(...position);
        // Initial angles are calculated to match the starting position
        this.orbitAngles = new Float32Array([
            Math.asin(position[2] / this.orbitDistance), // Pitch
            Math.atan2(position[1], position[0])         // Yaw
        ]);

        // Input state
        this.keys = new Set();
        this.isMouseDown = false;
        this.lastMousePos = { x: 0, y: 0 };
        this.lastTouchState = { dist: 0 };

        this._setInitialFreeCamRotation();
        this.addEventListeners();
    }

    // --- Math Helpers ---
    _v3Add(a, b) { return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]; }
    _v3Scale(v, s) { return [v[0] * s, v[1] * s, v[2] * s]; }
    _quatNormalize(q) {
        const len = Math.hypot(...q);
        if (len > 0) {
            q[0] /= len; q[1] /= len; q[2] /= len; q[3] /= len;
        }
        return q;
    }
    // ... (rest of the math helpers from previous version) ...
    _v3Subtract(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
    _v3Cross(a, b) { return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]; }
    _v3Normalize(v) {
        const len = Math.hypot(...v);
        if (len === 0) return [0, 0, 0];
        return [v[0] / len, v[1] / len, v[2] / len];
    }
    quatFromAxisAngle(axis, angle) {
        const halfAngle = angle * 0.5;
        const s = Math.sin(halfAngle);
        return new Float32Array([axis[0] * s, axis[1] * s, axis[2] * s, Math.cos(halfAngle)]);
    }
    quatMultiply(a, b) { // equivalent to: b * a
        const ax = a[0], ay = a[1], az = a[2], aw = a[3];
        const bx = b[0], by = b[1], bz = b[2], bw = b[3];
        return new Float32Array([
            ax * bw + aw * bx + ay * bz - az * by,
            ay * bw + aw * by + az * bx - ax * bz,
            az * bw + aw * bz + ax * by - ay * bx,
            aw * bw - ax * bx - ay * by - az * bz,
        ]);
    }
    vec3TransformQuat(v, q) {
        let uvx = q[1] * v[2] - q[2] * v[1], uvy = q[2] * v[0] - q[0] * v[2], uvz = q[0] * v[1] - q[1] * v[0];
        let uuvx = q[1] * uvz - q[2] * uvy, uuvy = q[2] * uvx - q[0] * uvz, uuvz = q[0] * uvy - q[1] * uvx;
        let w2 = q[3] * 2;
        uvx *= w2; uvy *= w2; uvz *= w2;
        uuvx *= 2; uuvy *= 2; uuvz *= 2;
        return [v[0] + uvx + uuvx, v[1] + uvy + uuvy, v[2] + uvz + uuvz];
    }
    mat4FromQuat(q) {
        let x = q[0], y = q[1], z = q[2], w = q[3];
        let x2 = x + x, y2 = y + y, z2 = z + z;
        let xx = x * x2, xy = x * y2, xz = x * z2;
        let yy = y * y2, yz = y * z2, zz = z * z2;
        let wx = w * x2, wy = w * y2, wz = w * z2;
        let out = new Float32Array(16);
        out[0] = 1 - (yy + zz); out[1] = xy + wz; out[2] = xz - wy; out[3] = 0;
        out[4] = xy - wz; out[5] = 1 - (xx + zz); out[6] = yz + wx; out[7] = 0;
        out[8] = xz + wy; out[9] = yz - wx; out[10] = 1 - (xx + yy); out[11] = 0;
        out[12] = 0; out[13] = 0; out[14] = 0; out[15] = 1;
        return out;
    }
    mat4LookAt(eye, center, up) {
        let out = new Float32Array(16);
        const z_axis = this._v3Normalize(this._v3Subtract(eye, center));
        const x_axis = this._v3Normalize(this._v3Cross(up, z_axis));
        const y_axis = this._v3Cross(z_axis, x_axis);
        out[0] = x_axis[0]; out[1] = y_axis[0]; out[2] = z_axis[0]; out[3] = 0;
        out[4] = x_axis[1]; out[5] = y_axis[1]; out[6] = z_axis[1]; out[7] = 0;
        out[8] = x_axis[2]; out[9] = y_axis[2]; out[10] = z_axis[2]; out[11] = 0;
        out[12] = -(x_axis[0] * eye[0] + x_axis[1] * eye[1] + x_axis[2] * eye[2]);
        out[13] = -(y_axis[0] * eye[0] + y_axis[1] * eye[1] + y_axis[2] * eye[2]);
        out[14] = -(z_axis[0] * eye[0] + z_axis[1] * eye[1] + z_axis[2] * eye[2]);
        out[15] = 1;
        return out;
    }
    mat4Perspective(fovy, aspect, near, far) {
        const f = 1.0 / Math.tan(fovy / 2);
        let out = new Float32Array(16);
        out[0] = f / aspect; out[5] = f; out[11] = -1; out[15] = 0;
        if (far != null && far !== Infinity) {
            const nf = 1 / (near - far);
            out[10] = (far + near) * nf;
            out[14] = 2 * far * near * nf;
        } else {
            out[10] = -1;
            out[14] = -2 * near;
        }
        return out;
    }

    // --- Core Logic ---
    _setInitialFreeCamRotation() {
        const tempView = this.mat4LookAt(this.position, this.orbitTarget, this.upAxis);
        const tempWorld = new Float32Array(16); // This would be mat4Invert(tempView)
        // Simplified inversion for lookAt (orthonormal rotation part)
        tempWorld[0] = tempView[0]; tempWorld[1] = tempView[4]; tempWorld[2] = tempView[8];
        tempWorld[4] = tempView[1]; tempWorld[5] = tempView[5]; tempWorld[6] = tempView[9];
        tempWorld[8] = tempView[2]; tempWorld[9] = tempView[6]; tempWorld[10]= tempView[10];

        // Decompose rotation matrix to quaternion
        const trace = tempWorld[0] + tempWorld[5] + tempWorld[10];
        if (trace > 0) {
            let S = Math.sqrt(trace + 1.0) * 2;
            this.rotation[3] = 0.25 * S;
            this.rotation[0] = (tempWorld[6] - tempWorld[9]) / S;
            this.rotation[1] = (tempWorld[8] - tempWorld[2]) / S;
            this.rotation[2] = (tempWorld[1] - tempWorld[4]) / S;
        } else if ((tempWorld[0] > tempWorld[5]) && (tempWorld[0] > tempWorld[10])) {
            let S = Math.sqrt(1.0 + tempWorld[0] - tempWorld[5] - tempWorld[10]) * 2;
            this.rotation[3] = (tempWorld[6] - tempWorld[9]) / S;
            this.rotation[0] = 0.25 * S;
            this.rotation[1] = (tempWorld[1] + tempWorld[4]) / S;
            this.rotation[2] = (tempWorld[8] + tempWorld[2]) / S;
        } else if (tempWorld[5] > tempWorld[10]) {
            let S = Math.sqrt(1.0 + tempWorld[5] - tempWorld[0] - tempWorld[10]) * 2;
            this.rotation[3] = (tempWorld[8] - tempWorld[2]) / S;
            this.rotation[0] = (tempWorld[1] + tempWorld[4]) / S;
            this.rotation[1] = 0.25 * S;
            this.rotation[2] = (tempWorld[6] + tempWorld[9]) / S;
        } else {
            let S = Math.sqrt(1.0 + tempWorld[10] - tempWorld[0] - tempWorld[5]) * 2;
            this.rotation[3] = (tempWorld[1] - tempWorld[4]) / S;
            this.rotation[0] = (tempWorld[8] + tempWorld[2]) / S;
            this.rotation[1] = (tempWorld[6] + tempWorld[9]) / S;
            this.rotation[2] = 0.25 * S;
        }
    }

    getViewMatrix() {
        if (this.mode === 'free') {
            // Build view matrix from position and quaternion: V = R_inv * T_inv
            const conjugateQuat = [ -this.rotation[0], -this.rotation[1], -this.rotation[2], this.rotation[3] ];
            const rotationInvMatrix = this.mat4FromQuat(conjugateQuat);

            const translationInv = [-this.position[0], -this.position[1], -this.position[2]];

            // Manually multiply R_inv * T_inv
            const viewMatrix = rotationInvMatrix;
            viewMatrix[12] = translationInv[0] * viewMatrix[0] + translationInv[1] * viewMatrix[4] + translationInv[2] * viewMatrix[8];
            viewMatrix[13] = translationInv[0] * viewMatrix[1] + translationInv[1] * viewMatrix[5] + translationInv[2] * viewMatrix[9];
            viewMatrix[14] = translationInv[0] * viewMatrix[2] + translationInv[1] * viewMatrix[6] + translationInv[2] * viewMatrix[10];

            return viewMatrix;
        } else { // 'orbit' mode
            const D = this.orbitDistance;
            const yaw = this.orbitAngles[1];
            const pitch = this.orbitAngles[0];
            const T = this.orbitTarget;

            const eye_x = T[0] + D * Math.cos(pitch) * Math.cos(yaw);
            const eye_y = T[1] + D * Math.cos(pitch) * Math.sin(yaw);
            const eye_z = T[2] + D * Math.sin(pitch);

            const eye = [eye_x, eye_y, eye_z];

            // Update the camera's canonical position property
            this.position[0] = eye[0];
            this.position[1] = eye[1];
            this.position[2] = eye[2];

            // Now create the view matrix from that position
            return this.mat4LookAt(eye, this.orbitTarget, this.upAxis);
        }
    }

    getProjectionMatrix(aspect) {
        return this.mat4Perspective(this.fovY * Math.PI / 180, aspect, this.nearZ, this.farZ);
    }

    update(dt) {
        if (this.mode !== 'free') return;

        const speed = this.baseMoveSpeed * (this.keys.has('ControlLeft') ? 3 : 1);
        let move = [0, 0, 0];

        if (this.keys.has('KeyW')) move[2] -= 1;
        if (this.keys.has('KeyS')) move[2] += 1;
        if (this.keys.has('KeyA')) move[0] -= 1;
        if (this.keys.has('KeyD')) move[0] += 1;
        if (this.keys.has('Space')) move[1] += 1;
        if (this.keys.has('ShiftLeft')) move[1] -= 1;

        const moveLength = Math.hypot(...move);
        if (moveLength > 0.001) {
            const normalizedMove = this._v3Scale(move, 1 / moveLength);
            const rotatedMove = this.vec3TransformQuat(normalizedMove, this.rotation);
            this.position = this._v3Add(this.position, this._v3Scale(rotatedMove, speed * dt));
        }

        // Handle roll
        let rollDelta = 0;
        if (this.keys.has('KeyQ')) rollDelta += this.rollSpeed * dt;
        if (this.keys.has('KeyE')) rollDelta -= this.rollSpeed * dt;

        if (Math.abs(rollDelta) > 0.001) {
            const rollQuat = this.quatFromAxisAngle([0, 0, 1], rollDelta); // Roll around local Z
            this.rotation = this.quatMultiply(this.rotation, rollQuat);
            this._quatNormalize(this.rotation);
        }
    }

    _handlePointerMove(x, y, isMouseDown) {
        if (!isMouseDown) return;

        const dx = x - this.lastMousePos.x;
        const dy = y - this.lastMousePos.y;
        this.lastMousePos = { x, y };

        const sensitivity = 0.004;

        if (this.mode === 'free') {
            const yawDelta = -dx * sensitivity;
            const pitchDelta = -dy * sensitivity;

            // Create delta rotations around local axes for a fully ego-centric feel.
            const yawQuat = this.quatFromAxisAngle([0, 1, 0], yawDelta);     // Yaw around local-Y (up)
            const pitchQuat = this.quatFromAxisAngle([1, 0, 0], pitchDelta);   // Pitch around local-X (right)

            // Combine the deltas (pitch, then yaw)
            const deltaQuat = this.quatMultiply(yawQuat, pitchQuat);

            // Apply the combined delta rotation in local space (post-multiplication)
            // newRotation = oldRotation * deltaRotation
            this.rotation = this.quatMultiply(this.rotation, deltaQuat);
            this._quatNormalize(this.rotation);

        } else { // Orbit mode
            this.orbitAngles[1] -= dx * sensitivity; // Yaw
            this.orbitAngles[0] += dy * sensitivity; // Pitch

            // Clamp pitch
            const PI_2 = Math.PI / 2;
            this.orbitAngles[0] = Math.max(-PI_2 + 0.001, Math.min(PI_2 - 0.001, this.orbitAngles[0]));
        }
    }
    addEventListeners() {
        this.canvas.addEventListener('mousedown', (e) => {
            if (e.button === 0) {
                this.isMouseDown = true;
                this.lastMousePos = { x: e.clientX, y: e.clientY };
                e.preventDefault();
            }
        });
        document.addEventListener('mouseup', (e) => {
            if (e.button === 0) this.isMouseDown = false;
        });
        document.addEventListener('mousemove', (e) => {
            this._handlePointerMove(e.clientX, e.clientY, this.isMouseDown);
        });

        window.addEventListener('keydown', (e) => {
            if (e.code === 'KeyM') {
                this.mode = (this.mode === 'free') ? 'orbit' : 'free';
                console.log(`Camera mode switched to: ${this.mode}`);
            }
            this.keys.add(e.code);
        });
        window.addEventListener('keyup', (e) => this.keys.delete(e.code));

        this.canvas.addEventListener('wheel', (e) => {
            e.preventDefault();
            if (this.mode === 'orbit') {
                this.orbitDistance = Math.max(0.1, this.orbitDistance + e.deltaY * 0.01);
            }
        }, { passive: false });

        this.canvas.addEventListener('touchstart', (e) => {
            e.preventDefault();
            this.isMouseDown = true;
            const touches = e.touches;
            if (touches.length === 1) {
                this.lastMousePos = { x: touches[0].clientX, y: touches[0].clientY };
            } else if (touches.length === 2) {
                this.lastTouchState.dist = Math.hypot(
                    touches[0].clientX - touches[1].clientX,
                    touches[0].clientY - touches[1].clientY
                );
            }
        }, { passive: false });

        this.canvas.addEventListener('touchend', (e) => {
            if (e.touches.length === 0) this.isMouseDown = false;
        });

        this.canvas.addEventListener('touchmove', (e) => {
            e.preventDefault();
            if (this.mode === 'orbit' && e.touches.length > 0) {
                if (e.touches.length === 1) {
                    this._handlePointerMove(e.touches[0].clientX, e.touches[0].clientY, true);
                } else if (e.touches.length === 2) {
                    const newDist = Math.hypot(
                        e.touches[0].clientX - e.touches[1].clientX,
                        e.touches[0].clientY - e.touches[1].clientY
                    );
                    const scale = this.lastTouchState.dist / newDist;
                    this.orbitDistance = Math.max(0.1, this.orbitDistance * scale);
                    this.lastTouchState.dist = newDist;
                }
            }
        }, { passive: false });

        this.canvas.addEventListener('contextmenu', e => e.preventDefault());
    }
}

// --- Web Worker Logic ---
function createWorker(self) {
    let data = { vertexCount: 0, tetCount: 0, vertices: null, indices: null, densities: null, colors: null, gradients: null, circumcenters: null, circumradiiSq: null };

    // NEW/CHANGED: ping-pong pool + reusable scratch
    const outBufferPool = [];              // buffers returned from main
    let sizeList = null;                   // Int32Array(M), reused
    let counts0 = null, starts0 = null;    // Uint32Array(65536), reused

    // Statically allocated buffers for sorting (unchanged)
    let keys, payload, tempKeys, tempPayload, intKeys;

    function processTetrahedralData(arrayBuffer) {
        const header = new Uint32Array(arrayBuffer, 0, 3);
        data.vertexCount = header[0];
        data.tetCount = header[1];
        data.shDegree = header[2];
        let offset = 4 * 3;

        const vertexByteLength = data.vertexCount * 3 * 4;
        data.vertices = new Float32Array(arrayBuffer.slice(offset, offset + vertexByteLength));
        offset += vertexByteLength;

        const indexByteLength = data.tetCount * 4 * 4;
        data.indices = new Uint32Array(arrayBuffer.slice(offset, offset + indexByteLength));
        offset += indexByteLength;

        const densityByteLength = data.tetCount * 1;
        data.densities = new Uint8Array(arrayBuffer.slice(offset, offset + densityByteLength));
        offset += densityByteLength;
        offset += (4 - (offset % 4)) % 4;

        const coeffsPerChannel = (data.shDegree + 1) * (data.shDegree + 1);
        const shByteLength = data.tetCount * coeffsPerChannel * 3; 
        
        data.shCoeffs = new Uint8Array(arrayBuffer.slice(offset, offset + shByteLength));
        offset += shByteLength;
        offset += (4 - (offset % 4)) % 4;


        const gradientByteLength = data.tetCount * 3 * 2;
        data.gradients = new Uint16Array(arrayBuffer.slice(offset, offset + gradientByteLength));
        // offset += gradientByteLength; // No longer needed if this is the last read

        // --- REPLACED SECTION START ---
        // Instead of reading circumcenters, we allocate array to fill later
        data.circumcenters = new Float32Array(data.tetCount * 3);
        data.circumradiiSq = new Float32Array(data.tetCount);
        // --- REPLACED SECTION END ---

        keys = new Float32Array(data.tetCount);
        intKeys     = new Uint32Array(data.tetCount);
        tempKeys    = new Uint32Array(data.tetCount);
        tempPayload = new Uint32Array(data.tetCount);

        // Loop over tetrahedra to calculate circumcenters and radii
        for (let i = 0; i < data.tetCount; i++) {
            // 1. Get Indices of the 4 vertices
            const i0 = data.indices[i * 4];
            const i1 = data.indices[i * 4 + 1];
            const i2 = data.indices[i * 4 + 2];
            const i3 = data.indices[i * 4 + 3];

            // 2. Get Coordinates for v0, v1, v2, v3
            const v0x = data.vertices[i0 * 3], v0y = data.vertices[i0 * 3 + 1], v0z = data.vertices[i0 * 3 + 2];
            const v1x = data.vertices[i1 * 3], v1y = data.vertices[i1 * 3 + 1], v1z = data.vertices[i1 * 3 + 2];
            const v2x = data.vertices[i2 * 3], v2y = data.vertices[i2 * 3 + 1], v2z = data.vertices[i2 * 3 + 2];
            const v3x = data.vertices[i3 * 3], v3y = data.vertices[i3 * 3 + 1], v3z = data.vertices[i3 * 3 + 2];

            // 3. Compute vectors relative to v0 (Python: a, b, c)
            const ax = v1x - v0x, ay = v1y - v0y, az = v1z - v0z;
            const bx = v2x - v0x, by = v2y - v0y, bz = v2z - v0z;
            const cx = v3x - v0x, cy = v3y - v0y, cz = v3z - v0z;

            // 4. Compute squares of lengths (Python: aa, bb, cc)
            const aa = ax * ax + ay * ay + az * az;
            const bb = bx * bx + by * by + bz * bz;
            const cc = cx * cx + cy * cy + cz * cz;

            // 5. Compute Cross Products
            // cross_bc = b x c
            const bcx = by * cz - bz * cy;
            const bcy = bz * cx - bx * cz;
            const bcz = bx * cy - by * cx;

            // cross_ca = c x a
            const cax = cy * az - cz * ay;
            const cay = cz * ax - cx * az;
            const caz = cx * ay - cy * ax;

            // cross_ab = a x b
            const abx = ay * bz - az * by;
            const aby = az * bx - ax * bz;
            const abz = ax * by - ay * bx;

            // 6. Compute Denominator (2 * dot(a, cross_bc))
            let denominator = 2.0 * (ax * bcx + ay * bcy + az * bcz);

            // Handle small denominator (degenerate tetrahedra)
            if (Math.abs(denominator) < 1e-12) {
                denominator = 1.0; 
            }

            // 7. Compute Relative Circumcenter
            // (aa * cross_bc + bb * cross_ca + cc * cross_ab) / denominator
            const rx = (aa * bcx + bb * cax + cc * abx) / denominator;
            const ry = (aa * bcy + bb * cay + cc * aby) / denominator;
            const rz = (aa * bcz + bb * caz + cc * abz) / denominator;

            // 8. Store Absolute Position (v0 + relative)
            data.circumcenters[i * 3]     = v0x + rx;
            data.circumcenters[i * 3 + 1] = v0y + ry;
            data.circumcenters[i * 3 + 2] = v0z + rz;

            // 9. Store Radius Squared (|relative|^2)
            // Python used linalg.norm, here we need squared for the shader usually
            data.circumradiiSq[i] = rx * rx + ry * ry + rz * rz;
        }

        sizeList = new Int32Array(data.tetCount);
        counts0 = new Uint32Array(256 * 256);
        starts0 = new Uint32Array(256 * 256);
    }

    self.onmessage = (e) => {
        if (e.data.returnBuffer) {
            outBufferPool.push(e.data.returnBuffer);
        } else if (e.data.fileBuffer) {
            processTetrahedralData(e.data.fileBuffer);
            self.postMessage({
                vertices: data.vertices,
                indices: data.indices,
                densities: data.densities,
                gradients: data.gradients,
                tetCount: data.tetCount,
                shCoeffs: data.shCoeffs,
                circumcenters: data.circumcenters,
                circumradiiSq: data.circumradiiSq,
                shDegree: data.shDegree,
            }, [
                data.densities.buffer, 
                data.gradients.buffer, 
                data.shCoeffs.buffer,
                data.circumcenters.buffer,
                data.circumradiiSq.buffer
            ]);
        }
    };
}


// --- WebGPU Main Application ---
async function main() {
    if (!navigator.gpu) {
        alert("WebGPU not supported on this browser.");
        return;
    }

    if (typeof RadixSort === 'undefined') {
        alert("RadixSort library not found. Ensure script tag is included.");
        return;
    }
    const { RadixSortKernel } = RadixSort;

    const canvas = document.getElementById("canvas");
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    
    // Check if adapter exists
    if (!adapter) {
        alert("No appropriate GPUAdapter found.");
        return;
    }

    const deviceLimits = {
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        maxBufferSize: adapter.limits.maxBufferSize,
        maxStorageBuffersPerShaderStage: Math.max(10, adapter.limits.maxStorageBuffersPerShaderStage),
    };

    const device = await adapter.requestDevice({
        requiredLimits: deviceLimits,
    });
    const context = canvas.getContext("webgpu");

    const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
    context.configure({
        device,
        format: presentationFormat,
        alphaMode: "premultiplied",
    });

    const fpsEl = document.getElementById("fps");
    const tetsDrawnEl = document.getElementById("tets-drawn");
    const messageEl = document.getElementById("message");
    const spinnerEl = document.getElementById("spinner");

    // --- Upscaling: dropdown UI (mobile-friendly) ------------------------------
    const UPSCALE_CHOICES = [16, 4, 2, 1];

    function getSavedUpscale() {
      const saved = Number(localStorage.getItem('upscaleFactor'));
      return UPSCALE_CHOICES.includes(saved) ? saved : 1; // default 2x
    }
    function saveUpscale(v) { localStorage.setItem('upscaleFactor', String(v)); }

    // Create a small overlay with a <select>
    function makeUpscaleDropdown(initial) {
      const wrap = document.createElement('div');
      wrap.style.position = 'fixed';
      wrap.style.top = '12px';
      wrap.style.right = '12px';
      wrap.style.zIndex = '1000';
      wrap.style.background = 'rgba(0,0,0,0.6)';
      wrap.style.color = '#fff';
      wrap.style.padding = '8px 10px';
      wrap.style.borderRadius = '8px';
      wrap.style.font = '12px system-ui, -apple-system, Segoe UI, Roboto, sans-serif';
      wrap.style.display = 'flex';
      wrap.style.gap = '8px';
      wrap.style.alignItems = 'center';
      wrap.style.backdropFilter = 'blur(6px)';

      const label = document.createElement('label');
      label.textContent = 'Upscale';
      label.htmlFor = 'upscaleSelect';

      const select = document.createElement('select');
      select.id = 'upscaleSelect';
      select.style.color = '#000';
      select.style.borderRadius = '6px';
      select.style.padding = '3px 6px';

      [['16','16x (fastest)'], ['4','4x (fastest)'], ['2','2x'], ['1','1x (native)']].forEach(([v, txt]) => {
        const opt = document.createElement('option');
        opt.value = v; opt.textContent = txt;
        select.appendChild(opt);
      });
      select.value = String(initial);

      // Prevent canvas handlers from grabbing these events
      ['pointerdown','mousedown','touchstart','wheel','click'].forEach(ev =>
        wrap.addEventListener(ev, e => e.stopPropagation(), { passive: false })
      );

      select.addEventListener('change', () => {
        upscaleFactor = Number(select.value);
        saveUpscale(upscaleFactor);
        resize(); // reallocate the backing buffer to new scale
        messageEl.innerText = `Left-drag to orbit. Press M to enable WASDQE to move.`;
      });

      wrap.append(label, select);
      document.body.appendChild(wrap);
      return select;
    }

    // Init value + UI
    let upscaleFactor = getSavedUpscale();
    const upscaleSelectEl = makeUpscaleDropdown(upscaleFactor);

    const shaderModule = device.createShaderModule({ label: "Volume Shader", code: wgslSource });
    const computeShaderModule = device.createShaderModule({ label: "Compute Shader", code: computeSource });

    function createStorageBuffer(typedArray) {
        // Align size to 4 bytes
        const size = (typedArray.byteLength + 3) & ~3; 
        const buffer = device.createBuffer({
            size: size,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true
        });
        const mappedRange = buffer.getMappedRange();
        
        // Copy data
        if (typedArray instanceof Uint32Array) new Uint32Array(mappedRange).set(typedArray);
        else if (typedArray instanceof Uint16Array) new Uint16Array(mappedRange).set(typedArray);
        else if (typedArray instanceof Uint8Array) new Uint8Array(mappedRange).set(typedArray);
        else if (typedArray instanceof Float32Array) new Float32Array(mappedRange).set(typedArray);
        
        buffer.unmap();
        return buffer;
    }

    // --- Layouts ---
    const renderBindGroupLayout = device.createBindGroupLayout({
        label: "Render Layout",
        entries: [
            { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
            { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
            { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
            { binding: 3, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
            { binding: 4, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
            { binding: 5, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
            { binding: 6, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
        ]
    });

    const computeBindGroupLayout = device.createBindGroupLayout({
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
            { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
            { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
            { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
            { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
            { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
            { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
            { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
            { binding: 10, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        ]
    });

    const pipeline = device.createRenderPipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [renderBindGroupLayout] }),
        vertex: { module: shaderModule, entryPoint: "vs_main" },
        fragment: {
            module: shaderModule,
            entryPoint: "fs_main",
            targets: [{
                format: presentationFormat,
                blend: {
                    color: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
                    alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" }
                }
            }]
        },
        primitive: { topology: "triangle-list", cullMode: "back" }
    });

    const computePipeline = device.createComputePipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [computeBindGroupLayout] }),
        compute: { module: computeShaderModule, entryPoint: "main" },
    });

    // --- Variables ---
    let vertexBuffer, indexBuffer, densityBuffer, colorBuffer, gradientBuffer, sortedIndexBuffer;
    let circumCenterBuffer, circumRadiiBuffer, keysBuffer, valuesBuffer, indirectBuffer;
    let bindGroup, computeBindGroup;
    let radixSorter;
    let tetCount = 0;
    let shDegree = 0;
    let visibleCount = 0;

    const uniformBufferSize = 112; 
    const uniformBuffer = device.createBuffer({
        size: uniformBufferSize,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    const cpuUniformBuffer = new ArrayBuffer(uniformBufferSize);
    const uniformFloats = new Float32Array(cpuUniformBuffer);
    const uniformUints = new Uint32Array(cpuUniformBuffer);

    // --- Worker Setup ---
    const worker = new Worker(URL.createObjectURL(new Blob([`(${createWorker.toString()})(self)`], { type: "application/javascript" })));

    worker.onmessage = (e) => {
        if (e.data.vertices) {
            spinnerEl.style.display = 'none';
            messageEl.innerText = 'Left-drag to look. Press M to enable WASDQE.';
            tetCount = e.data.tetCount;
            shDegree = e.data.shDegree;
            
            // <--- FIX: Using the scoped helper function without passing device
            vertexBuffer = createStorageBuffer(e.data.vertices);
            indexBuffer = createStorageBuffer(e.data.indices);

            // Densities processing
            const denRaw = e.data.densities;
            const denF32 = new Float32Array(denRaw.length);
            for (let i = 0; i < denRaw.length; i++) {
                denF32[i] = Math.exp((denRaw[i]-100)/20);
            }
            densityBuffer = createStorageBuffer(denF32);

            // Colors setup
            const colorData = new Float32Array(e.data.tetCount * 3);
            colorBuffer = device.createBuffer({
                size: colorData.byteLength,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
            });

            // Gradients processing
            const gradRaw = e.data.gradients;
            const gradF32 = new Float32Array(gradRaw.length);
            for (let i = 0; i < gradRaw.length; i++) {
                gradF32[i] = decodeHalf(gradRaw[i]);
            }
            gradientBuffer = createStorageBuffer(gradF32);

            const shBuffer = createStorageBuffer(e.data.shCoeffs);
            circumCenterBuffer = createStorageBuffer(e.data.circumcenters);
            circumRadiiBuffer = createStorageBuffer(e.data.circumradiiSq);

            // buffers for sort/indirect
            sortedIndexBuffer = device.createBuffer({ size: tetCount * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
            keysBuffer = device.createBuffer({ size: tetCount * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
            valuesBuffer = device.createBuffer({ size: tetCount * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
            indirectBuffer = device.createBuffer({ size: 4 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC});

            computeBindGroup = device.createBindGroup({
                layout: computeBindGroupLayout,
                entries: [
                    { binding: 0, resource: { buffer: uniformBuffer } },
                    { binding: 1, resource: { buffer: vertexBuffer } },
                    { binding: 2, resource: { buffer: indexBuffer } },
                    { binding: 3, resource: { buffer: colorBuffer } }, 
                    { binding: 4, resource: { buffer: gradientBuffer } },
                    { binding: 5, resource: { buffer: shBuffer } },
                    { binding: 6, resource: { buffer: circumCenterBuffer } },
                    { binding: 7, resource: { buffer: circumRadiiBuffer } },
                    { binding: 8, resource: { buffer: keysBuffer } },
                    { binding: 9, resource: { buffer: valuesBuffer } },
                    { binding: 10, resource: { buffer: indirectBuffer } },
                ]
            });

            radixSorter = new RadixSortKernel({
                device,
                keys: keysBuffer,
                values: valuesBuffer,
                count: tetCount,
                check_order: false,
                bit_count: 32,                    // Number of bits per element. Must be a multiple of 4 (default: 32)
                workgroup_size: { x: 16, y: 16 }, // Workgroup size in x and y dimensions. (x * y) must be a power of two
            });

            createBindGroup();

        } else if (e.data.depthIndex && sortedIndexBuffer) {
            visibleCount = e.data.visibleCount;
            device.queue.writeBuffer(sortedIndexBuffer, 0, e.data.depthIndex, 0, visibleCount);
            worker.postMessage({ returnBuffer: e.data.depthIndex.buffer }, [e.data.depthIndex.buffer]);
        }
    };

    function createBindGroup() {
        if (!vertexBuffer) return;
        bindGroup = device.createBindGroup({
            layout: renderBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: uniformBuffer } },
                { binding: 1, resource: { buffer: vertexBuffer } },
                { binding: 2, resource: { buffer: indexBuffer } },
                { binding: 3, resource: { buffer: densityBuffer } },
                { binding: 4, resource: { buffer: colorBuffer } },
                { binding: 5, resource: { buffer: gradientBuffer } },
                { binding: 6, resource: { buffer: valuesBuffer } },
            ]
        });
    }

    // --- Rendering ---
    const camera = new Camera([0, 3, -3], canvas);
    
    const resize = () => {
        const cssW = window.innerWidth;
        const cssH = window.innerHeight;
        canvas.style.width = cssW + 'px';
        canvas.style.height = cssH + 'px';
        const dpr = (window.devicePixelRatio || 1) / upscaleFactor;
        canvas.width = Math.round(cssW * dpr);
        canvas.height = Math.round(cssH * dpr);
    };
    window.addEventListener("resize", resize);
    resize();

    let frameCount = 0;
    let lastHudTime = 0;
    let lastFrameTime = performance.now();
    const resetIndirectCmd = new Uint32Array([12, 0, 0, 0]);

    const frame = (now) => {
        if (!indirectBuffer || !uniformBuffer || !bindGroup) {
            requestAnimationFrame(frame);
            return;
        }

        const dt = (now - lastFrameTime) / 1000.0;
        lastFrameTime = now;

        // FPS Logic
        frameCount++;
        const timeSinceHud = now - lastHudTime;
        if (timeSinceHud >= 1000) {
            const fps = Math.round((frameCount * 1000) / timeSinceHud);
            fpsEl.innerText = `${fps} fps`;
            tetsDrawnEl.innerText = `${tetCount.toLocaleString()} tets`;
            frameCount = 0;
            lastHudTime = now;
        }

        camera.update(dt);
        const aspect = canvas.width / canvas.height;
        const viewMat = camera.getViewMatrix();
        const projMat = camera.getProjectionMatrix(aspect);
        const viewProj = multiply4(projMat, viewMat); 

        // Write Uniforms
        uniformFloats.set(viewProj, 0);
        uniformFloats[16] = camera.position[0];
        uniformFloats[17] = camera.position[1];
        uniformFloats[18] = camera.position[2];
        uniformFloats[20] = camera.position[0];
        uniformFloats[21] = camera.position[1];
        uniformFloats[22] = camera.position[2];
        uniformFloats[23] = canvas.width;
        uniformFloats[24] = canvas.height;
        uniformUints[25] = tetCount;
        uniformUints[26] = shDegree;
        
        device.queue.writeBuffer(uniformBuffer, 0, cpuUniformBuffer);
        device.queue.writeBuffer(indirectBuffer, 0, resetIndirectCmd);

        const commandEncoder = device.createCommandEncoder();

        // --- COMPUTE PASS ---
        if (computeBindGroup) {
            const pass = commandEncoder.beginComputePass();
            pass.setPipeline(computePipeline);
            pass.setBindGroup(0, computeBindGroup);
            
            // Dispatch 1 thread per tet
            const workgroupSize = 64;
            const totalWorkgroups = Math.ceil(tetCount / workgroupSize);
            const maxPerDim = 65535;
            const dispatchX = Math.min(totalWorkgroups, maxPerDim);
            const dispatchY = Math.ceil(totalWorkgroups / maxPerDim);
            console.log("Running", dispatchX, dispatchY);
            pass.dispatchWorkgroups(dispatchX, dispatchY);
            pass.end();
        }

        // --- SORT PASS ---
        const sortPass = commandEncoder.beginComputePass({ label: "Radix Sort" });
        radixSorter.dispatch(sortPass);
        sortPass.end();

        // --- RENDER PASS ---
        const passEncoder = commandEncoder.beginRenderPass({
            colorAttachments: [{
                view: context.getCurrentTexture().createView(),
                clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
                loadOp: "clear",
                storeOp: "store",
            }]
        });

        if (bindGroup) {
            passEncoder.setPipeline(pipeline);
            passEncoder.setBindGroup(0, bindGroup);
            // <--- FIX: Variable name mismatch. Was 'renderPass', changed to 'passEncoder'
            passEncoder.drawIndirect(indirectBuffer, 0); 
        }

        passEncoder.end();
        device.queue.submit([commandEncoder.finish()]);

        requestAnimationFrame(frame);
    };

    async function loadFile(url) {
        spinnerEl.style.display = 'block';
        messageEl.innerText = `Downloading ${url}...`;
        try {
            const req = await fetch(url);
            if (!req.ok) throw new Error(`HTTP error! status: ${req.status}`);
            const compressedBuffer = await req.arrayBuffer();
            const arrayBuffer = pako.inflate(new Uint8Array(compressedBuffer)).buffer;
            worker.postMessage({ fileBuffer: arrayBuffer }, [arrayBuffer]);
        } catch (error) {
            console.error('Failed to load file:', error);
            messageEl.innerText = `Could not load file.`;
            spinnerEl.style.display = 'none';
        }
    }

    // Drag and Drop
    const preventDefault = (e) => { e.preventDefault(); e.stopPropagation(); };
    document.addEventListener('dragenter', preventDefault);
    document.addEventListener('dragover', preventDefault);
    document.addEventListener('dragleave', preventDefault);
    document.addEventListener('drop', (e) => {
        messageEl.innerText = 'Loading...';
        preventDefault(e);
        const file = e.dataTransfer.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (event) => {
                spinnerEl.style.display = 'block';
                messageEl.innerText = 'Processing data...';
                
                try {
                    // 1. Get the compressed data
                    const compressedBuffer = event.target.result;
                    
                    // 2. Inflate using Pako (same logic as loadFile)
                    const decompressedData = pako.inflate(new Uint8Array(compressedBuffer));
                    
                    // 3. Extract the raw ArrayBuffer from the Uint8Array
                    const arrayBuffer = decompressedData.buffer;

                    // 4. Send the decompressed buffer to the worker
                    worker.postMessage({ fileBuffer: arrayBuffer }, [arrayBuffer]);
                } catch (error) {
                    console.error('Decompression failed:', error);
                    messageEl.innerText = 'Error: Invalid or corrupt file.';
                    spinnerEl.style.display = 'none';
                }
            };
            reader.readAsArrayBuffer(file);
        }
    });

    const urlParams = new URLSearchParams(window.location.search);
    const fileNameFromUrl = urlParams.get('file');

    // Check if a 'file' parameter was provided in the URL
    if (fileNameFromUrl) {
        const base_url = `https://pub-51212dd9bd624f0baa58d43d062bc53f.r2.dev/r1.0/`
        const fileToLoad = base_url + `${urlParams.get('file')}` || "corsair.rmesh";
        console.log(`Loading file from URL parameter: ${fileToLoad}`);
        loadFile(fileToLoad);
    }
    
    requestAnimationFrame(frame);
}

// Helper to launch
main().catch(console.error);

