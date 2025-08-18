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


class Camera {
    constructor(position = [0, 5, 2], canvas) {
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

    let camPos;
    let lastPos = [];

    // Statically allocated buffers for sorting (unchanged)
    let keys, payload, tempKeys, tempPayload, intKeys;

    // NEW: get an output array from pool or allocate once
    function getDepthIndexArray(len) {
        for (let i = 0; i < outBufferPool.length; i++) {
            const buf = outBufferPool[i];
            if (buf.byteLength === len * 4) { // Uint32Array length match
                outBufferPool.splice(i, 1);
                return new Uint32Array(buf);
            }
        }
        return new Uint32Array(len);
    }

    var _floatView = new Float32Array(1);
    var _int32View = new Int32Array(_floatView.buffer);
    function floatToHalf(float) {
        _floatView[0] = float;
        var f = _int32View[0];

        var sign = (f >> 31) & 0x0001;
        var exp = (f >> 23) & 0x00ff;
        var frac = f & 0x007fffff;

        var newExp;
        if (exp == 0) {
            newExp = 0;
        } else if (exp < 113) {
            newExp = 0;
            frac |= 0x00800000;
            frac = frac >> (113 - exp);
            if (frac & 0x01000000) {
                newExp = 1;
                frac = 0;
            }
        } else if (exp < 142) {
            newExp = exp - 112;
        } else {
            newExp = 31;
            frac = 0;
        }

        return (sign << 15) | (newExp << 10) | (frac >> 13);
    }

    const float32ToUint16 = (() => {
      const buf = new ArrayBuffer(4);
      const f32 = new Float32Array(buf);
      const u32 = new Uint32Array(buf);

      return (v) => {
        f32[0] = v;
        const x = u32[0];

        const sign = (x >>> 16) & 0x8000;      // 1 bit
        let   exp  = (x >>> 23) & 0xff;        // 8 bits
        let   mant =  x & 0x7fffff;            // 23 bits

        // Inf / NaN
        if (exp === 0xff) {
          return sign | (mant ? 0x7e00 : 0x7c00); // qNaN or Inf
        }

        // Zero / subnormal in f32: treat as zero in half (too small to matter)
        if (exp === 0 && mant === 0) return sign;

        // Re-bias exponent from f32 (127) to f16 (15)
        let e16 = exp - 127 + 15;

        // Overflow -> Inf
        if (e16 >= 31) return sign | 0x7c00;

        if (e16 <= 0) {
          // Subnormal half: fold implicit 1, shift, and RNE
          if (e16 < -10) return sign; // too small -> ±0
          mant |= 0x00800000; // add hidden 1
          // shift so that we’ll end with a 10-bit mantissa after rounding
          const shift = 14 - e16;               // 1..10
          // guard+round+sticky over 13 bits total to end at 10 bits
          let m = mant >>> (shift + 13);
          const rem = mant & ((1 << (shift + 13)) - 1);
          const halfway = 1 << (shift + 12);
          // round-to-nearest-even
          if (rem > halfway || (rem === halfway && (m & 1))) m += 1;
          return sign | m;
        }

        // Normal half: round 23->10 bits (keep guard/round/sticky)
        // remainder below bit 13
        const rem = mant & 0x1fff;
        let m10 = mant >>> 13;
        const halfway = 0x1000; // 1<<12
        if (rem > halfway || (rem === halfway && (m10 & 1))) {
          m10 += 1;
          if (m10 === 0x400) {   // mantissa overflow -> bump exponent
            m10 = 0;
            e16 += 1;
            if (e16 >= 31) return sign | 0x7c00; // became Inf
          }
        }
        return sign | (e16 << 10) | m10;
      };
    })();
    // --- NEW: sort mode ('half16' | 'f32x2') ------------------------------------
    // let sortMode = 'half16';
    let sortMode = 'f32x2';

    // float -> sortable uint32 (ascending numeric order)
    const f32ToSortableU32 = (() => {
      const buf = new ArrayBuffer(4);
      const f32 = new Float32Array(buf);
      const u32 = new Uint32Array(buf);
      return (v) => {
        f32[0] = v;
        let u = u32[0] >>> 0;
        // if negative: invert all bits; else flip sign bit
        u = (u & 0x80000000) ? (~u >>> 0) : (u ^ 0x80000000);
        return u >>> 0;
      };
    })();

    // one 16-bit LSD counting pass (stable)
    function radix16Pass(inKeys, inPayload, outKeys, outPayload, shift) {
      // buckets are 0..65535, reuse global counts0/starts0
      counts0.fill(0);
      for (let i = 0, M = inKeys.length; i < M; i++) {
        counts0[(inKeys[i] >>> shift) & 0xFFFF]++;
      }
      starts0[0] = 0;
      for (let b = 1; b < 65536; b++) starts0[b] = starts0[b - 1] + counts0[b - 1];

      for (let i = 0, M = inKeys.length; i < M; i++) {
        const b = (inKeys[i] >>> shift) & 0xFFFF;
        const dst = starts0[b]++;
        outKeys[dst] = inKeys[i];
        outPayload[dst] = inPayload[i];
      }
    }

    function processTetrahedralData(arrayBuffer) {
        const header = new Uint32Array(arrayBuffer, 0, 2);
        data.vertexCount = header[0];
        data.tetCount = header[1];
        let offset = 8;

        const vertexByteLength = data.vertexCount * 3 * 4;
        data.vertices = new Float32Array(arrayBuffer.slice(offset, offset + vertexByteLength));
        offset += vertexByteLength;

        const indexByteLength = data.tetCount * 4 * 4;
        data.indices = new Uint32Array(arrayBuffer.slice(offset, offset + indexByteLength));
        offset += indexByteLength;

        const densityByteLength = data.tetCount * 2;
        data.densities = new Uint16Array(arrayBuffer.slice(offset, offset + densityByteLength));
        offset += densityByteLength;

        const colorByteLength = data.tetCount * 3 * 2;
        data.colors = new Uint16Array(arrayBuffer.slice(offset, offset + colorByteLength));
        offset += colorByteLength;

        const gradientByteLength = data.tetCount * 3 * 2;
        data.gradients = new Uint16Array(arrayBuffer.slice(offset, offset + gradientByteLength));
        offset += gradientByteLength;

        const circumcenterByteLength = data.tetCount * 3 * 4;
        data.circumcenters = new Float32Array(arrayBuffer.slice(offset, offset + circumcenterByteLength));

        data.circumradiiSq = new Float32Array(data.tetCount);

        keys = new Float32Array(data.tetCount);
        payload = new Uint32Array(data.tetCount);
        for (let i = 0; i < data.tetCount; i++) payload[i] = i;
        intKeys     = new Uint32Array(data.tetCount);
        tempKeys    = new Uint32Array(data.tetCount);
        tempPayload = new Uint32Array(data.tetCount);

        for (let i = 0; i < data.tetCount; i++) {
            const v_idx = data.indices[i * 4];
            const vx = data.vertices[v_idx * 3], vy = data.vertices[v_idx * 3 + 1], vz = data.vertices[v_idx * 3 + 2];
            const cx = data.circumcenters[i * 3], cy = data.circumcenters[i * 3 + 1], cz = data.circumcenters[i * 3 + 2];
            const dx = vx - cx, dy = vy - cy, dz = vz - cz;
            data.circumradiiSq[i] = dx * dx + dy * dy + dz * dz;
        }

        // NEW: allocate reusable scratch now that tetCount is known
        sizeList = new Int32Array(data.tetCount);
        counts0 = new Uint32Array(256 * 256);
        starts0 = new Uint32Array(256 * 256);
    }

    /**
     * Recalculates keys, culls primitives, and sorts the payload buffer.
     */
    function cullAndSortTets(camPos) {
        if (!data.vertices) return;
        x = camPos[0] - lastPos[0]
        y = camPos[1] - lastPos[1]
        z = camPos[2] - lastPos[2]
        if ((x*x + y*y + z*z) < 0.01) return;
        console.log(sortMode);
        console.time("sort");


        const M = data.tetCount; // CHANGED: make constant & explicit
        let maxDepth = -Infinity;
        let minDepth = Infinity;

        // (Re)use arrays
        if (!sizeList || sizeList.length !== M) sizeList = new Int32Array(M);
        counts0.fill(0);
        starts0.fill(0);
        if (sortMode === 'f32x2') {
          // --- Float32 keys + two 16-bit passes (stable) ---------------------------
          for (let i = 0; i < M; i++) {
            const cx = data.circumcenters[i * 3], cy = data.circumcenters[i * 3 + 1], cz = data.circumcenters[i * 3 + 2];
            const dx = cx - camPos[0], dy = cy - camPos[1], dz = cz - camPos[2];
            const floatValue = -(dx * dx + dy * dy + dz * dz - data.circumradiiSq[i]);
            intKeys[i] = f32ToSortableU32(floatValue);
            // payload[] was initialized to i in processTetrahedralData()
          }
          for (let i = 0; i < M; i++) payload[i] = i;

          // two LSD passes: low 16 -> temp, high 16 -> back
          radix16Pass(intKeys, payload, tempKeys, tempPayload, 0);
          radix16Pass(tempKeys, tempPayload, intKeys, payload, 16);

          // Ship out a pooled copy of the sorted payload
          const depthIndex = getDepthIndexArray(M);
          depthIndex.set(payload);
          console.timeEnd("sort");
          lastPos = camPos;
          self.postMessage({ depthIndex, M }, [depthIndex.buffer]);

        } else {
          // --- Existing half16 single-pass path (unchanged) -------------------------
          let maxDepth = -Infinity, minDepth = Infinity;

          for (let i = 0; i < M; i++) {
            const cx = data.circumcenters[i * 3], cy = data.circumcenters[i * 3 + 1], cz = data.circumcenters[i * 3 + 2];
            const dx = cx - camPos[0], dy = cy - camPos[1], dz = cz - camPos[2];
            let floatValue = -(dx * dx + dy * dy + dz * dz - data.circumradiiSq[i]);

            const u16 = float32ToUint16(floatValue);
            let depth = (u16 & 0x8000) ? ((~u16) & 0xFFFF) : (u16 ^ 0x8000);
            sizeList[i] = depth;
            if (depth > maxDepth) maxDepth = depth;
            if (depth < minDepth) minDepth = depth;
          }

          const range = maxDepth - minDepth || 1;
          const depthInv = (65536 - 1) / range;
          counts0.fill(0);
          for (let i = 0; i < M; i++) {
            sizeList[i] = ((sizeList[i] - minDepth) * depthInv) | 0;
            counts0[sizeList[i]]++;
          }
          starts0[0] = 0;
          for (let i = 1; i < 65536; i++) starts0[i] = starts0[i - 1] + counts0[i - 1];

          const depthIndex = getDepthIndexArray(M);
          for (let i = 0; i < M; i++) depthIndex[starts0[sizeList[i]]++] = i;

          console.timeEnd("sort");
          lastPos = camPos;
          self.postMessage({ depthIndex, M }, [depthIndex.buffer]);
        }
        /*

        for (let i = 0; i < M; i++) {
            const cx = data.circumcenters[i * 3], cy = data.circumcenters[i * 3 + 1], cz = data.circumcenters[i * 3 + 2];
            const dx = cx - camPos[0], dy = cy - camPos[1], dz = cz - camPos[2];

            let floatValue = -(dx * dx + dy * dy + dz * dz - data.circumradiiSq[i]);
            const uint16Value = float32ToUint16(floatValue);
            // const uint16Value = floatToHalf(floatValue);

            let depth;
            if (uint16Value & 0x8000) {
                depth = (~uint16Value) & 0xFFFF;
            } else {
                depth = uint16Value ^ 0x8000;
            }
            sizeList[i] = depth;
            if (depth > maxDepth) maxDepth = depth;
            if (depth < minDepth) minDepth = depth;
        }

        // 16-bit single-pass counting sort
        const range = maxDepth - minDepth || 1;              // avoid div by zero
        const depthInv = (256 * 256 - 1) / range;
        for (let i = 0; i < M; i++) {
            sizeList[i] = ((sizeList[i] - minDepth) * depthInv) | 0;
            counts0[sizeList[i]]++;
        }
        for (let i = 1; i < 256 * 256; i++) starts0[i] = starts0[i - 1] + counts0[i - 1];

        const depthIndex = getDepthIndexArray(M);            // NEW: pooled buffer
        for (let i = 0; i < M; i++) depthIndex[starts0[sizeList[i]]++] = i;

        console.timeEnd("sort");

        lastPos = camPos;
        self.postMessage({ depthIndex, M }, [depthIndex.buffer]); // transfer out
        */
    }

    let sortRunning;

    const throttledSort = () => {
        if (!sortRunning) {
            sortRunning = true;
            const prev = camPos;
            cullAndSortTets(prev);
            setTimeout(() => {
                sortRunning = false;
                if (prev !== camPos) throttledSort();
            }, 0);
        }
    };

    self.onmessage = (e) => {
        if (e.data.returnBuffer) {
            // main thread returned ownership -> reuse next frame
            outBufferPool.push(e.data.returnBuffer); // NEW
        } else if (e.data.sortMode) {
            sortMode = e.data.sortMode;
        } else if (e.data.fileBuffer) {
            processTetrahedralData(e.data.fileBuffer);
            self.postMessage({
                vertices: data.vertices,
                indices: data.indices,
                densities: data.densities,
                colors: data.colors,
                gradients: data.gradients,
                tetCount: data.tetCount
            }, [data.vertices.buffer, data.indices.buffer, data.densities.buffer, data.colors.buffer, data.gradients.buffer]);
        } else if (e.data.camPos) {
            camPos = e.data.camPos;
            throttledSort();
        }
    };
}


// --- Main Application Logic ---
async function main() {
    const canvas = document.getElementById("canvas");
    // const gl = canvas.getContext("webgl2", { antialias: false });
    const gl = canvas.getContext("webgl2", {
      alpha: false,
      premultipliedAlpha: false,
      antialias: false,
      depth: false,
      stencil: false,
      preserveDrawingBuffer: false,
      desynchronized: true,
      powerPreference: 'high-performance'
    });
    if (!gl) { alert("WebGL 2 not supported!"); return; }
    halfFloatExt = gl.getExtension('OES_texture_half_float');

    const maxTexSize = gl.getParameter(gl.MAX_TEXTURE_SIZE);
    console.log("Max Texture Size:", maxTexSize);

    const fpsEl = document.getElementById("fps"), tetsDrawnEl = document.getElementById("tets-drawn"), messageEl = document.getElementById("message"), spinnerEl = document.getElementById("spinner");

    // --- Upscaling: dropdown UI (mobile-friendly) ------------------------------
    const UPSCALE_CHOICES = [16, 4, 2, 1];

    function getSavedUpscale() {
      const saved = Number(localStorage.getItem('upscaleFactor'));
      return UPSCALE_CHOICES.includes(saved) ? saved : 2; // default 2x
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
        messageEl.innerText = `Right-drag to look. WASDQE to move. (Press U to cycle)`;
      });

      wrap.append(label, select);
      document.body.appendChild(wrap);
      return select;
    }

    // Init value + UI
    let upscaleFactor = getSavedUpscale();
    const upscaleSelectEl = makeUpscaleDropdown(upscaleFactor);

    // Let users cycle with 'U' and keep dropdown in sync
    window.addEventListener('keydown', (e) => {
      if (e.code === 'KeyU') {
        upscaleFactor = (upscaleFactor === 4) ? 2 : (upscaleFactor === 2 ? 1 : 4);
        saveUpscale(upscaleFactor);
        if (upscaleSelectEl) upscaleSelectEl.value = String(upscaleFactor);
        resize();
        messageEl.innerText = `Upscaling: ${upscaleFactor}x — Right-drag to look. WASDQE to move. (Press U to cycle)`;
      }
    });

    // Optional: let users cycle at runtime (press 'U')
    window.addEventListener('keydown', (e) => {
        if (e.code === 'KeyU') {
            upscaleFactor = (upscaleFactor === 4) ? 2 : (upscaleFactor === 2 ? 1 : 4);
            localStorage.setItem('upscaleFactor', String(upscaleFactor));
            resize(); // reallocate the backing buffer
            messageEl.innerText = `Upscaling: ${upscaleFactor}x — Right-drag to look. WASDQE to move. (Press U to cycle)`;
        }
    });

    const vshaderFileResponse = await fetch("vs.glsl");
    if (!vshaderFileResponse.ok) {
        const errorMsg = "Failed to load shaders from shaders/raster.glsl";
        document.getElementById("message").innerText = errorMsg;
        throw new Error(errorMsg);
    }
    const vertexShaderSource = await vshaderFileResponse.text();
    const fshaderFileResponse = await fetch("fs.glsl");
    if (!fshaderFileResponse.ok) {
        const errorMsg = "Failed to load shaders from shaders/raster.glsl";
        document.getElementById("message").innerText = errorMsg;
        throw new Error(errorMsg);
    }
    const fragmentShaderSource = await fshaderFileResponse.text();

    const vertexShader = gl.createShader(gl.VERTEX_SHADER);
    gl.shaderSource(vertexShader, vertexShaderSource);
    gl.compileShader(vertexShader);
    if (!gl.getShaderParameter(vertexShader, gl.COMPILE_STATUS)) { console.error("VS Error:", gl.getShaderInfoLog(vertexShader)); return; }

    const fragmentShader = gl.createShader(gl.FRAGMENT_SHADER);
    gl.shaderSource(fragmentShader, fragmentShaderSource);
    gl.compileShader(fragmentShader);
    if (!gl.getShaderParameter(fragmentShader, gl.COMPILE_STATUS)) { console.error("FS Error:", gl.getShaderInfoLog(fragmentShader)); return; }

    const program = gl.createProgram();
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) { console.error("Link Error:", gl.getProgramInfoLog(program)); return; }
    gl.useProgram(program);

    // gl.enable(gl.PRIMITIVE_RESTART_FIXED_INDEX);
    gl.cullFace(gl.BACK);
    gl.disable(gl.DEPTH_TEST);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);

    const u_viewProjection = gl.getUniformLocation(program, "viewProjection");
    const u_rayOrigin = gl.getUniformLocation(program, "rayOrigin");
    const u_verticesTexture = gl.getUniformLocation(program, "verticesTexture");
    const u_indicesTexture = gl.getUniformLocation(program, "indicesTexture");
    const u_densityTexture = gl.getUniformLocation(program, "densityTexture");
    const u_colorTexture = gl.getUniformLocation(program, "colorTexture");
    const u_gradientTexture = gl.getUniformLocation(program, "gradientTexture");
    const u_verticesTextureSize = gl.getUniformLocation(program, "verticesTextureSize");
    const u_indicesTextureSize = gl.getUniformLocation(program, "indicesTextureSize");
    const u_densityTextureSize = gl.getUniformLocation(program, "densityTextureSize");
    const u_colorTextureSize = gl.getUniformLocation(program, "colorTextureSize");
    const u_gradientTextureSize = gl.getUniformLocation(program, "gradientTextureSize");

    const indexBuffer = gl.createBuffer();
    // This index data defines the 4 triangle faces using the 4 base vertices
    const indexData = new Uint16Array([
        0, 2, 1,   // Face 1
        1, 2, 3,   // Face 2
        0, 3, 2,   // Face 3
        3, 0, 1    // Face 4
        // 2, 3, 0, 1, 2, 3 // A different valid strip sequence
        // Strip 1 (2 faces)
        // 0, 1, 2, 3,
        // // Stop this strip and start a new one
        // 0xFFFF,
        // // Strip 2 (the other 2 faces)
        // 1, 0, 3, 2
    ]);
    // Bind it as the ELEMENT_ARRAY_BUFFER
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indexData, gl.STATIC_DRAW);

    // const vertexIdInTetBuffer = gl.createBuffer();
    // const vertexIdInTetData = new Uint8Array(Array.from({length: 4}, (_, i) => i));
    // gl.bindBuffer(gl.ARRAY_BUFFER, vertexIdInTetBuffer);
    // gl.bufferData(gl.ARRAY_BUFFER, vertexIdInTetData, gl.STATIC_DRAW);
    // const a_vertexIdInTet = gl.getAttribLocation(program, "vertexIdInTet");
    // gl.enableVertexAttribArray(a_vertexIdInTet);
    // gl.vertexAttribIPointer(a_vertexIdInTet, 1, gl.UNSIGNED_BYTE, 0, 0);

    const sortedIndexBuffer = gl.createBuffer();
    const a_tetId = gl.getAttribLocation(program, "tetId");
    gl.enableVertexAttribArray(a_tetId);
    gl.bindBuffer(gl.ARRAY_BUFFER, sortedIndexBuffer);
    gl.vertexAttribIPointer(a_tetId, 1, gl.UNSIGNED_INT, 0, 0);
    gl.vertexAttribDivisor(a_tetId, 1);

    const verticesTexture = gl.createTexture(), indicesTexture = gl.createTexture(), densityTexture = gl.createTexture(), colorTexture = gl.createTexture(), gradientTexture = gl.createTexture();

    let visibleCount = 0;
    const worker = new Worker(URL.createObjectURL(new Blob([`(${createWorker.toString()})(self)`], { type: "application/javascript" })));

    // NEW: pick sort mode via URL (?sort=f32 or ?sort=half) or default to 'half'
    const sortParam = (new URLSearchParams(window.location.search).get('sort') || '').toLowerCase();
    worker.postMessage({ sortMode: sortParam.startsWith('f32') ? 'f32x2' : 'half16' });

    // Optional: runtime toggle with 'R'
    window.addEventListener('keydown', (e) => {
      if (e.code === 'KeyR') {
        const next = (sortParam.startsWith('f32') || window.__sortMode === 'f32x2') ? 'half16' : 'f32x2';
        window.__sortMode = next;
        worker.postMessage({ sortMode: next });
        messageEl.innerText = `Sort mode: ${next === 'f32x2' ? 'float32 (2-pass)' : 'half16 (1-pass)'}`;
      }
    });

    worker.onmessage = (e) => {
        if (e.data.vertices) {
            spinnerEl.style.display = 'none';
            messageEl.innerText = 'Right-drag to look. WASDQE to move.';

            const setupTex = (tex, unit, loc, sizeLoc, internalFormat, format, type, data, elementsPerTexel) => {
                gl.activeTexture(gl.TEXTURE0 + unit);
                gl.bindTexture(gl.TEXTURE_2D, tex);

                const totalElements = data.length / elementsPerTexel;
                const width = Math.min(totalElements, maxTexSize);
                const height = Math.ceil(totalElements / width);

                const paddedSize = width * height * elementsPerTexel;
                let paddedData = data;
                if (data.length < paddedSize) {
                    paddedData = new data.constructor(paddedSize);
                    paddedData.set(data);
                }

                gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
                gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
                gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, width, height, 0, format, type, paddedData);
                gl.uniform1i(loc, unit);
                gl.uniform2i(sizeLoc, width, height);
            };

            setupTex(verticesTexture, 0, u_verticesTexture, u_verticesTextureSize, gl.RGB32F, gl.RGB, gl.FLOAT, e.data.vertices, 3);
            setupTex(indicesTexture, 1, u_indicesTexture, u_indicesTextureSize, gl.RGBA32UI, gl.RGBA_INTEGER, gl.UNSIGNED_INT, e.data.indices, 4);
            setupTex(densityTexture, 2, u_densityTexture, u_densityTextureSize, gl.R16F, gl.RED, gl.HALF_FLOAT, e.data.densities, 1);
            setupTex(colorTexture, 3, u_colorTexture, u_colorTextureSize, gl.RGB16F, gl.RGB, gl.HALF_FLOAT, e.data.colors, 3);
            setupTex(gradientTexture, 4, u_gradientTexture, u_gradientTextureSize, gl.RGB16F, gl.RGB, gl.HALF_FLOAT, e.data.gradients, 3);


        } else if (e.data.depthIndex) {
            visibleCount = e.data.M;
            gl.bindBuffer(gl.ARRAY_BUFFER, sortedIndexBuffer);
            gl.bufferData(gl.ARRAY_BUFFER, e.data.depthIndex, gl.DYNAMIC_DRAW);
            worker.postMessage({ returnBuffer: e.data.depthIndex.buffer }, [e.data.depthIndex.buffer]);
        }
        // lastSortedCameraPosition.set(camera.position);
    };

    const camera = new Camera([0, 0, 5], canvas);
    let lastFrameTime = 0;
    let avgFps = 0;
    let lastSortedCameraPosition = new Float32Array(3);
    let loaded = false;
    const SORT_TRIGGER_THRESHOLD_SQ = 0.01 * 0.01; // 1cm

    const resize = () => {
      // Fill window visually
      const cssW = window.innerWidth;
      const cssH = window.innerHeight;
      gl.canvas.style.width = cssW + 'px';
      gl.canvas.style.height = cssH + 'px';

      // Render at lower backing resolution
      const dpr = (window.devicePixelRatio || 1) / upscaleFactor;
      gl.canvas.width  = Math.round(cssW * dpr);
      gl.canvas.height = Math.round(cssH * dpr);

      gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
    };
    window.addEventListener("resize", resize);
    resize();

    let nextHud = 0;
    const HUD_INTERVAL_MS = 1000;

    const frame = (now) => {
        const dt = (now - lastFrameTime) / 1000.0;
        lastFrameTime = now;

          if (now >= nextHud) {
            const currentFps = 1 / Math.max(dt, 0.00001);
            avgFps = currentFps;
            fpsEl.innerText = `${Math.round(avgFps)} fps`;
            tetsDrawnEl.innerText = `${visibleCount.toLocaleString()} tetrahedra`;
            nextHud = now + HUD_INTERVAL_MS;
          }

        camera.update(dt);


        const viewMatrix = camera.getViewMatrix();
        const projectionMatrix = camera.getProjectionMatrix(gl.canvas.clientWidth / gl.canvas.clientHeight);
        const viewProj = multiply4(projectionMatrix, viewMatrix);

        worker.postMessage({ camPos: camera.position });

        gl.uniformMatrix4fv(u_viewProjection, false, viewProj);
        gl.uniform3fv(u_rayOrigin, camera.position);

        // gl.clearColor(0.1, 0.1, 0.15, 1.0);
        gl.clear(gl.COLOR_BUFFER_BIT);

        if (visibleCount > 0) {
            // gl.drawArraysInstanced(gl.TRIANGLES, 0, 12, visibleCount);
            gl.drawElementsInstanced(
                gl.TRIANGLES,      // mode
                12,                // count (number of indices to draw)
                gl.UNSIGNED_SHORT, // type of data in the index buffer
                0,                 // offset
                visibleCount
            );
        }
        requestAnimationFrame(frame);
    };
    async function loadFile(url) {
        spinnerEl.style.display = 'block';
        messageEl.innerText = `Downloading ${url}...`;
        try {
            const req = await fetch(url);
            if (!req.ok) {
                throw new Error(`HTTP error! status: ${req.status}`);
            }
            const arrayBuffer = await req.arrayBuffer();
            messageEl.innerText = 'Processing data...';
            worker.postMessage({ fileBuffer: arrayBuffer }, [arrayBuffer]);
            loaded = true;
        } catch (error) {
            console.error('Failed to load default file:', error);
            messageEl.innerText = `Could not load default file. Please drop a file.`;
            spinnerEl.style.display = 'none';
        }
    }


    const preventDefault = (e) => { e.preventDefault(); e.stopPropagation(); };
    document.addEventListener('dragenter', preventDefault);
    document.addEventListener('dragover', preventDefault);
    document.addEventListener('dragleave', preventDefault);
    document.addEventListener('drop', (e) => {
        preventDefault(e);
        const file = e.dataTransfer.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (event) => {
                spinnerEl.style.display = 'block';
                messageEl.innerText = 'Processing data...';
                worker.postMessage({ fileBuffer: event.target.result }, [event.target.result]);
                loaded = true;
            };
            reader.readAsArrayBuffer(file);
        }
    });
    const urlParams = new URLSearchParams(window.location.search);
    const fileNameFromUrl = urlParams.get('file');

    let fileToLoad;
    // Check if a 'file' parameter was provided in the URL
    if (fileNameFromUrl) {
        fileToLoad = `splats/${fileNameFromUrl}`;
        console.log(`Loading file from URL parameter: ${fileToLoad}`);
        loadFile(fileToLoad);
    } else {
        // Otherwise, fall back to the default file
        // fileToLoad = "splats/room_small.splat";
        // console.log(`Loading default file: ${fileToLoad}`);
    }


    frame(0);
}

main().catch(err => {
    console.error(err);
    document.getElementById("message").innerText = `Fatal Error: ${err.message}`;
    document.getElementById("spinner").style.display = 'none';
});
