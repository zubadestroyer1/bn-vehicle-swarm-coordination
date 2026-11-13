/*
 * Flagship Offline WebGL2 3D City & Neural Dynamic Bayesian Network Engine
 * Pure native WebGL2 + GLSL ES 3.00 — zero external dependencies, zero network calls.
 */
(function () {
  "use strict";

  const REL_COLORS = [
    [0.22, 0.74, 0.98], // 0: SAME_CORRIDOR (Cyan-Blue Platoon)
    [0.99, 0.42, 0.52], // 1: SAME_JUNCTION_CONFLICT (Coral-Rose Conflict)
    [0.98, 0.68, 0.16], // 2: DOWNSTREAM_SPILLBACK (Amber-Gold Spillback)
    [0.20, 0.85, 0.60], // 3: UPSTREAM_FEEDER (Emerald Green-Wave Feeder)
  ];

  const REL_NAMES = [
    "Same-Corridor Platoon",
    "Junction Conflict",
    "Downstream Spillback",
    "Upstream Feeder Wave",
  ];

  const MODE_NAMES = {
    "-1": "NATIVE_SIGNAL",
    "0": "GO_WAVE",
    "1": "CRUISE",
    "2": "ECO_GLIDE",
    "3": "YIELD",
  };

  // ===========================================================================
  // 1. Linear Algebra Helpers (Right-Handed Z-Up World: X=East, Y=North, Z=Up)
  // ===========================================================================

  function mat4Identity() {
    const m = new Float32Array(16);
    m[0] = 1; m[5] = 1; m[10] = 1; m[15] = 1;
    return m;
  }

  function mat4Multiply(a, b) {
    const out = new Float32Array(16);
    for (let c = 0; c < 4; c++) {
      const b0 = b[c * 4 + 0], b1 = b[c * 4 + 1], b2 = b[c * 4 + 2], b3 = b[c * 4 + 3];
      out[c * 4 + 0] = a[0] * b0 + a[4] * b1 + a[8] * b2 + a[12] * b3;
      out[c * 4 + 1] = a[1] * b0 + a[5] * b1 + a[9] * b2 + a[13] * b3;
      out[c * 4 + 2] = a[2] * b0 + a[6] * b1 + a[10] * b2 + a[14] * b3;
      out[c * 4 + 3] = a[3] * b0 + a[7] * b1 + a[11] * b2 + a[15] * b3;
    }
    return out;
  }

  function mat4Perspective(fovYRad, aspect, near, far) {
    const f = 1.0 / Math.tan(0.5 * fovYRad);
    const nf = 1.0 / (near - far);
    const out = new Float32Array(16);
    out[0] = f / Math.max(0.1, aspect);
    out[5] = f;
    out[10] = (far + near) * nf;
    out[11] = -1.0;
    out[14] = 2.0 * far * near * nf;
    return out;
  }

  function mat4Ortho(halfHeight, aspect, near, far) {
    const top = Math.max(5.0, halfHeight);
    const right = top * Math.max(0.1, aspect);
    const lr = 1.0 / right;
    const bt = 1.0 / top;
    const nf = 1.0 / (near - far);
    const out = new Float32Array(16);
    out[0] = lr;
    out[5] = bt;
    out[10] = 2.0 * nf;
    out[14] = (far + near) * nf;
    out[15] = 1.0;
    return out;
  }

  function mat4LookAtZUp(eye, target, upHint) {
    let fx = target[0] - eye[0];
    let fy = target[1] - eye[1];
    let fz = target[2] - eye[2];
    const flen = Math.hypot(fx, fy, fz) || 1.0;
    fx /= flen; fy /= flen; fz /= flen;

    let ux = upHint ? upHint[0] : 0.0;
    let uy = upHint ? upHint[1] : 0.0;
    let uz = upHint ? upHint[2] : 1.0;
    if (Math.abs(fx * ux + fy * uy + fz * uz) > 0.995) {
      ux = 0.0; uy = 1.0; uz = 0.0;
    }

    let sx = fy * uz - fz * uy;
    let sy = fz * ux - fx * uz;
    let sz = fx * uy - fy * ux;
    const slen = Math.hypot(sx, sy, sz) || 1.0;
    sx /= slen; sy /= slen; sz /= slen;

    const vx = sy * fz - sz * fy;
    const vy = sz * fx - sx * fz;
    const vz = sx * fy - sy * fx;

    const out = mat4Identity();
    out[0] = sx; out[4] = sy; out[8] = sz;
    out[1] = vx; out[5] = vy; out[9] = vz;
    out[2] = -fx; out[6] = -fy; out[10] = -fz;
    out[12] = -(sx * eye[0] + sy * eye[1] + sz * eye[2]);
    out[13] = -(vx * eye[0] + vy * eye[1] + vz * eye[2]);
    out[14] = fx * eye[0] + fy * eye[1] + fz * eye[2];
    return out;
  }

  function projectWorldToScreen(mvp, x, y, z, width, height) {
    const cx = mvp[0] * x + mvp[4] * y + mvp[8] * z + mvp[12];
    const cy = mvp[1] * x + mvp[5] * y + mvp[9] * z + mvp[13];
    const cw = mvp[3] * x + mvp[7] * y + mvp[11] * z + mvp[15];
    if (cw <= 0.05) {
      return null;
    }
    const ndcX = cx / cw;
    const ndcY = cy / cw;
    return [
      (ndcX * 0.5 + 0.5) * width,
      (0.5 - ndcY * 0.5) * height,
      cw,
    ];
  }

  function wrapAngle(rad) {
    let a = (rad + Math.PI) % (2.0 * Math.PI);
    if (a < 0) a += 2.0 * Math.PI;
    return a - Math.PI;
  }

  function hashColorFromId(str) {
    let h = 2166136261 >>> 0;
    const s = String(str || "goal");
    for (let i = 0; i < s.length; i++) {
      h ^= s.charCodeAt(i);
      h = Math.imul(h, 16777619) >>> 0;
    }
    const hue = (h % 360) / 360.0;
    return hslToRgb(hue, 0.76, 0.58);
  }

  function hslToRgb(h, s, l) {
    const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
    const p = 2 * l - q;
    const conv = (t) => {
      if (t < 0) t += 1;
      if (t > 1) t -= 1;
      if (t < 1 / 6) return p + (q - p) * 6 * t;
      if (t < 1 / 2) return q;
      if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
      return p;
    };
    return [conv(h + 1 / 3), conv(h), conv(h - 1 / 3)];
  }

  function speedRampRgb(v) {
    const t = Math.max(0.0, Math.min(1.0, v / 14.5));
    if (t < 0.45) {
      const u = t / 0.45;
      return [0.95 - 0.05 * u, 0.24 + 0.50 * u, 0.26 - 0.08 * u];
    }
    const u = (t - 0.45) / 0.55;
    return [0.90 - 0.72 * u, 0.74 + 0.14 * u, 0.18 + 0.52 * u];
  }

  function cascadeRampRgb(psi) {
    const t = Math.max(0.0, Math.min(1.0, psi / 2.5));
    return [0.22 + 0.76 * t, 0.52 - 0.12 * t + 0.36 * (1 - t), 0.94 - 0.68 * t];
  }

  function uncertaintyRampRgb(posStd) {
    const t = Math.max(0.0, Math.min(1.0, posStd / 4.5));
    return [0.20 + 0.72 * t, 0.78 - 0.50 * t, 0.72 + 0.24 * t];
  }

  // ===========================================================================
  // 2. GLSL ES 3.00 Shader Programs
  // ===========================================================================

  const LIT_VS = `#version 300 es
  precision highp float;
  layout(location = 0) in vec3 a_pos;
  layout(location = 1) in vec3 a_norm;
  layout(location = 2) in vec4 a_color;
  layout(location = 3) in vec2 a_meta;

  uniform mat4 u_mvp;
  out vec3 v_worldPos;
  out vec3 v_norm;
  out vec4 v_color;
  out vec2 v_meta;

  void main() {
    v_worldPos = a_pos;
    v_norm = a_norm;
    v_color = a_color;
    v_meta = a_meta;
    gl_Position = u_mvp * vec4(a_pos, 1.0);
  }`;

  const LIT_FS = `#version 300 es
  precision highp float;
  in vec3 v_worldPos;
  in vec3 v_norm;
  in vec4 v_color;
  in vec2 v_meta;

  uniform vec3 u_eyePos;
  uniform vec3 u_sceneCenter;
  uniform float u_fogScale;

  out vec4 fragColor;

  void main() {
    float matKind = v_meta.x;
    vec3 N = normalize(v_norm);
    vec3 L = normalize(vec3(0.46, -0.56, 0.69));
    vec3 V = normalize(u_eyePos - v_worldPos);
    vec3 H = normalize(L + V);

    vec3 base = v_color.rgb;

    if (matKind > 2.5) {
      // Emissive lamps, signals, brake lights, turn blinkers
      fragColor = vec4(base, v_color.a);
      return;
    }

    if (matKind > 0.5 && matKind < 1.5) {
      // Architectural building facade procedural window mullions & ground AO
      float z = v_worldPos.z;
      float u = v_meta.y;
      if (abs(N.z) < 0.35 && z > 2.2) {
        float floorBand = fract(z / 3.5);
        float colBand = fract(u / 3.2);
        float isGlass = step(0.24, floorBand) * step(floorBand, 0.78) *
                        step(0.18, colBand) * step(colBand, 0.82);
        vec3 glassTint = mix(vec3(0.14, 0.21, 0.30), vec3(0.38, 0.52, 0.68), clamp(dot(N, V), 0.0, 1.0));
        float warmLit = step(0.83, fract(sin(floor(z / 3.5) * 12.9898 + floor(u / 3.2) * 78.233) * 43758.5453));
        glassTint = mix(glassTint, vec3(0.92, 0.79, 0.52), warmLit * 0.28);
        base = mix(base, glassTint, isGlass * 0.72);
      }
      float ao = mix(0.68, 1.0, clamp(z / 9.0, 0.0, 1.0));
      base *= ao;
    }

    float ndotl = max(dot(N, L), 0.0);
    float hemi = 0.5 * N.z + 0.5;
    vec3 ambient = mix(vec3(0.22, 0.24, 0.29), vec3(0.46, 0.52, 0.62), hemi);
    vec3 lit = base * (ambient + 0.78 * ndotl);

    if (matKind > 1.5 && matKind < 2.5) {
      // Vehicle clearcoat specular sheen
      float spec = pow(max(dot(N, H), 0.0), 36.0);
      lit += vec3(0.32, 0.35, 0.40) * spec;
    }

    // Subtle atmospheric haze far from scene center
    float dist = length(v_worldPos.xy - u_sceneCenter.xy);
    float fog = clamp((dist - u_fogScale * 0.75) / max(120.0, u_fogScale * 0.85), 0.0, 0.48);
    lit = mix(lit, vec3(0.07, 0.10, 0.15), fog);

    fragColor = vec4(lit, v_color.a);
  }`;

  const ARC_VS = `#version 300 es
  precision highp float;
  layout(location = 0) in vec3 a_pos;
  layout(location = 1) in vec2 a_uv;
  layout(location = 2) in vec4 a_color;
  layout(location = 3) in vec2 a_pulse;

  uniform mat4 u_mvp;
  out vec2 v_uv;
  out vec4 v_color;
  out vec2 v_pulse;

  void main() {
    v_uv = a_uv;
    v_color = a_color;
    v_pulse = a_pulse;
    gl_Position = u_mvp * vec4(a_pos, 1.0);
  }`;

  const ARC_FS = `#version 300 es
  precision highp float;
  in vec2 v_uv;
  in vec4 v_color;
  in vec2 v_pulse;

  uniform float u_animTime;
  out vec4 fragColor;

  void main() {
    float mode = v_pulse.y;
    if (mode < -0.5) {
      // Soft radial glow ring (Kalman uncertainty halo / spillback overlay)
      float edgeSoft = 1.0 - smoothstep(0.55, 1.0, abs(v_uv.y));
      fragColor = vec4(v_color.rgb, v_color.a * edgeSoft);
      return;
    }

    // Glowing 3D DBN Dependency Arc with directional causal pulse src(u=0) -> dst(u=1)
    float crossFade = exp(-2.8 * v_uv.y * v_uv.y);
    float endFade = smoothstep(0.0, 0.07, v_uv.x) * (1.0 - smoothstep(0.93, 1.0, v_uv.x));
    float speed = 0.85 + 2.1 * clamp(v_pulse.y, 0.0, 1.0);
    float phase = fract(v_uv.x * 1.85 - u_animTime * speed);
    float wave = pow(max(0.0, 1.0 - abs(phase - 0.5) * 3.4), 2.4);

    vec3 col = mix(v_color.rgb, vec3(1.0, 0.99, 0.96), 0.58 * wave);
    float alpha = v_color.a * crossFade * (0.42 + 0.58 * wave) * (0.35 + 0.65 * endFade);
    fragColor = vec4(col, clamp(alpha, 0.0, 1.0));
  }`;

  function compileShader(gl, type, src) {
    const sh = gl.createShader(type);
    gl.shaderSource(sh, src);
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
      const info = gl.getShaderInfoLog(sh);
      gl.deleteShader(sh);
      throw new Error("WebGL2 shader compile error: " + info);
    }
    return sh;
  }

  function createProgram(gl, vsSrc, fsSrc) {
    const vs = compileShader(gl, gl.VERTEX_SHADER, vsSrc);
    const fs = compileShader(gl, gl.FRAGMENT_SHADER, fsSrc);
    const prog = gl.createProgram();
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    gl.deleteShader(vs);
    gl.deleteShader(fs);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      const info = gl.getProgramInfoLog(prog);
      gl.deleteProgram(prog);
      throw new Error("WebGL2 program link error: " + info);
    }
    return prog;
  }

  // ===========================================================================
  // 3. Mesh Builder Buffer (12 floats/vertex: pos3, norm3, col4, meta2)
  // ===========================================================================

  class LitMeshBuilder {
    constructor(initialFloats) {
      this.data = new Float32Array(initialFloats || 65536);
      this.len = 0;
    }

    clear() {
      this.len = 0;
    }

    ensure(extraFloats) {
      if (this.len + extraFloats <= this.data.length) return;
      let nextCap = this.data.length * 2;
      while (nextCap < this.len + extraFloats) nextCap *= 2;
      const grown = new Float32Array(nextCap);
      grown.set(this.data.subarray(0, this.len));
      this.data = grown;
    }

    pushVertex(x, y, z, nx, ny, nz, r, g, b, a, m0, m1) {
      this.ensure(12);
      const d = this.data;
      let o = this.len;
      d[o++] = x; d[o++] = y; d[o++] = z;
      d[o++] = nx; d[o++] = ny; d[o++] = nz;
      d[o++] = r; d[o++] = g; d[o++] = b; d[o++] = a;
      d[o++] = m0; d[o++] = m1;
      this.len = o;
    }

    pushQuad(p0, p1, p2, p3, n, col, m0, u0, u1) {
      const r = col[0], g = col[1], b = col[2], a = col.length > 3 ? col[3] : 1.0;
      const uSpan = u1 !== undefined ? u1 : 0.0;
      const uStart = u0 !== undefined ? u0 : 0.0;
      this.pushVertex(p0[0], p0[1], p0[2], n[0], n[1], n[2], r, g, b, a, m0, uStart);
      this.pushVertex(p1[0], p1[1], p1[2], n[0], n[1], n[2], r, g, b, a, m0, uSpan);
      this.pushVertex(p2[0], p2[1], p2[2], n[0], n[1], n[2], r, g, b, a, m0, uSpan);

      this.pushVertex(p0[0], p0[1], p0[2], n[0], n[1], n[2], r, g, b, a, m0, uStart);
      this.pushVertex(p2[0], p2[1], p2[2], n[0], n[1], n[2], r, g, b, a, m0, uSpan);
      this.pushVertex(p3[0], p3[1], p3[2], n[0], n[1], n[2], r, g, b, a, m0, uStart);
    }

    pushRibbon2D(pts, width, z, col, m0) {
      if (!pts || pts.length < 2) return;
      const hw = 0.5 * width;
      for (let i = 0; i < pts.length - 1; i++) {
        const x0 = pts[i][0], y0 = pts[i][1];
        const x1 = pts[i + 1][0], y1 = pts[i + 1][1];
        const dx = x1 - x0, dy = y1 - y0;
        const segLen = Math.hypot(dx, dy);
        if (segLen < 1e-4) continue;
        const nx = (-dy / segLen) * hw;
        const ny = (dx / segLen) * hw;
        this.pushQuad(
          [x0 - nx, y0 - ny, z],
          [x1 - nx, y1 - ny, z],
          [x1 + nx, y1 + ny, z],
          [x0 + nx, y0 + ny, z],
          [0, 0, 1],
          col,
          m0 || 0.0,
          0.0,
          0.0
        );
      }
    }

    pushDashedRibbon2D(pts, width, dashLen, gapLen, z, col) {
      if (!pts || pts.length < 2) return;
      const cycle = dashLen + gapLen;
      let distAcc = 0.0;
      const hw = 0.5 * width;
      for (let i = 0; i < pts.length - 1; i++) {
        const x0 = pts[i][0], y0 = pts[i][1];
        const x1 = pts[i + 1][0], y1 = pts[i + 1][1];
        const dx = x1 - x0, dy = y1 - y0;
        const segLen = Math.hypot(dx, dy);
        if (segLen < 1e-3) continue;
        const ux = dx / segLen, uy = dy / segLen;
        const nx = -uy * hw, ny = ux * hw;
        let cursor = 0.0;
        while (cursor < segLen) {
          const inCycle = (distAcc + cursor) % cycle;
          if (inCycle < dashLen) {
            const take = Math.min(dashLen - inCycle, segLen - cursor);
            const sx = x0 + ux * cursor, sy = y0 + uy * cursor;
            const ex = sx + ux * take, ey = sy + uy * take;
            this.pushQuad(
              [sx - nx, sy - ny, z],
              [ex - nx, ey - ny, z],
              [ex + nx, ey + ny, z],
              [sx + nx, sy + ny, z],
              [0, 0, 1],
              col,
              0.0,
              0.0,
              0.0
            );
            cursor += take;
          } else {
            cursor += Math.min(cycle - inCycle, segLen - cursor);
          }
        }
        distAcc += segLen;
      }
    }

    pushOrientedBox(cx, cy, z0, z1, yaw, halfLen, halfWid, col, m0, topScale) {
      const c = Math.cos(yaw), s = Math.sin(yaw);
      const tScale = topScale !== undefined ? topScale : 1.0;
      const hl0 = halfLen, hw0 = halfWid;
      const hl1 = halfLen * tScale, hw1 = halfWid * tScale;

      const worldPt = (lx, ly, z) => [cx + lx * c - ly * s, cy + lx * s + ly * c, z];

      const b0 = worldPt(-hl0, -hw0, z0);
      const b1 = worldPt(hl0, -hw0, z0);
      const b2 = worldPt(hl0, hw0, z0);
      const b3 = worldPt(-hl0, hw0, z0);

      const t0 = worldPt(-hl1, -hw1, z1);
      const t1 = worldPt(hl1, -hw1, z1);
      const t2 = worldPt(hl1, hw1, z1);
      const t3 = worldPt(-hl1, hw1, z1);

      // Top face
      this.pushQuad(t0, t1, t2, t3, [0, 0, 1], col, m0, 0, 0);
      // Front (+X local)
      this.pushQuad(b1, b2, t2, t1, [c, s, 0.15], col, m0, 0, 0);
      // Rear (-X local)
      this.pushQuad(b3, b0, t0, t3, [-c, -s, 0.15], col, m0, 0, 0);
      // Right (-Y local)
      this.pushQuad(b0, b1, t1, t0, [s, -c, 0.15], col, m0, 0, 0);
      // Left (+Y local)
      this.pushQuad(b2, b3, t3, t2, [-s, c, 0.15], col, m0, 0, 0);
    }

    get vertexCount() {
      return (this.len / 12) | 0;
    }
  }

  // Arc/Overlay mesh builder (11 floats/vertex: pos3, uv2, col4, pulse2)
  class ArcMeshBuilder {
    constructor(initialFloats) {
      this.data = new Float32Array(initialFloats || 32768);
      this.len = 0;
    }

    clear() {
      this.len = 0;
    }

    ensure(extraFloats) {
      if (this.len + extraFloats <= this.data.length) return;
      let nextCap = this.data.length * 2;
      while (nextCap < this.len + extraFloats) nextCap *= 2;
      const grown = new Float32Array(nextCap);
      grown.set(this.data.subarray(0, this.len));
      this.data = grown;
    }

    pushVertex(x, y, z, u, v, r, g, b, a, p0, p1) {
      this.ensure(11);
      const d = this.data;
      let o = this.len;
      d[o++] = x; d[o++] = y; d[o++] = z;
      d[o++] = u; d[o++] = v;
      d[o++] = r; d[o++] = g; d[o++] = b; d[o++] = a;
      d[o++] = p0; d[o++] = p1;
      this.len = o;
    }

    pushQuadUV(p0, p1, p2, p3, u0, u1, col, pulse0, pulse1) {
      const r = col[0], g = col[1], b = col[2], a = col[3];
      this.pushVertex(p0[0], p0[1], p0[2], u0, -1.0, r, g, b, a, pulse0, pulse1);
      this.pushVertex(p1[0], p1[1], p1[2], u1, -1.0, r, g, b, a, pulse0, pulse1);
      this.pushVertex(p2[0], p2[1], p2[2], u1, 1.0, r, g, b, a, pulse0, pulse1);

      this.pushVertex(p0[0], p0[1], p0[2], u0, -1.0, r, g, b, a, pulse0, pulse1);
      this.pushVertex(p2[0], p2[1], p2[2], u1, 1.0, r, g, b, a, pulse0, pulse1);
      this.pushVertex(p3[0], p3[1], p3[2], u0, 1.0, r, g, b, a, pulse0, pulse1);
    }

    pushRing2D(cx, cy, z, innerR, outerR, segments, col) {
      for (let s = 0; s < segments; s++) {
        const a0 = (s / segments) * Math.PI * 2.0;
        const a1 = ((s + 1) / segments) * Math.PI * 2.0;
        const c0 = Math.cos(a0), s0 = Math.sin(a0);
        const c1 = Math.cos(a1), s1 = Math.sin(a1);
        this.pushQuadUV(
          [cx + c0 * innerR, cy + s0 * innerR, z],
          [cx + c1 * innerR, cy + s1 * innerR, z],
          [cx + c1 * outerR, cy + s1 * outerR, z],
          [cx + c0 * outerR, cy + s0 * outerR, z],
          s / segments,
          (s + 1) / segments,
          col,
          1.0,
          -1.0
        );
      }
    }

    pushParabolicArc3D(x0, y0, x1, y1, zBase, apexH, ribbonHalfW, col, omega, gamma, eyePos) {
      const dx = x1 - x0, dy = y1 - y0;
      const dist = Math.hypot(dx, dy);
      if (dist < 0.8) return;
      const steps = dist > 90.0 ? 18 : 12;

      const sample = (u) => {
        const px = x0 + dx * u;
        const py = y0 + dy * u;
        const pz = zBase + 4.0 * apexH * u * (1.0 - u);
        return [px, py, pz];
      };

      let prevPt = sample(0.0);
      for (let k = 1; k <= steps; k++) {
        const u0 = (k - 1) / steps;
        const u1 = k / steps;
        const currPt = sample(u1);

        // Camera-facing billboard normal crossed with 3D arc tangent so ribbon is thick from EVERY angle
        const tx = currPt[0] - prevPt[0];
        const ty = currPt[1] - prevPt[1];
        const tz = currPt[2] - prevPt[2];
        const vx = eyePos[0] - currPt[0];
        const vy = eyePos[1] - currPt[1];
        const vz = eyePos[2] - currPt[2];
        let bx = ty * vz - tz * vy;
        let by = tz * vx - tx * vz;
        let bz = tx * vy - ty * vx;
        const blen = Math.hypot(bx, by, bz) || 1.0;
        bx = (bx / blen) * ribbonHalfW;
        by = (by / blen) * ribbonHalfW;
        bz = (bz / blen) * ribbonHalfW;

        this.pushQuadUV(
          [prevPt[0] - bx, prevPt[1] - by, prevPt[2] - bz],
          [currPt[0] - bx, currPt[1] - by, currPt[2] - bz],
          [currPt[0] + bx, currPt[1] + by, currPt[2] + bz],
          [prevPt[0] + bx, prevPt[1] + by, prevPt[2] + bz],
          u0,
          u1,
          col,
          omega,
          gamma
        );
        prevPt = currPt;
      }
    }

    get vertexCount() {
      return (this.len / 11) | 0;
    }
  }

  // ===========================================================================
  // 4. Static Urban Scene Compilation (Roads, Markings, Signals, 3D Buildings)
  // ===========================================================================

  function buildStaticCityBuffers(scene) {
    const roadBuilder = new LitMeshBuilder(262144);
    const buildingBuilder = new LitMeshBuilder(262144);

    const b = scene.bounds;
    const pad = Math.max(120.0, 0.35 * Math.max(b.span_x, b.span_y));
    const gx0 = b.min_x - pad, gx1 = b.max_x + pad;
    const gy0 = b.min_y - pad, gy1 = b.max_y + pad;

    // 1. Ground urban slate baseplate + 50m survey grid lines
    roadBuilder.pushQuad(
      [gx0, gy0, -0.06],
      [gx1, gy0, -0.06],
      [gx1, gy1, -0.06],
      [gx0, gy1, -0.06],
      [0, 0, 1],
      [0.09, 0.12, 0.16, 1.0],
      0.0
    );

    const gridStep = 50.0;
    const gridCol = [0.13, 0.17, 0.22, 1.0];
    for (let x = Math.ceil(gx0 / gridStep) * gridStep; x <= gx1; x += gridStep) {
      roadBuilder.pushRibbon2D([[x, gy0], [x, gy1]], 0.28, -0.03, gridCol, 0.0);
    }
    for (let y = Math.ceil(gy0 / gridStep) * gridStep; y <= gy1; y += gridStep) {
      roadBuilder.pushRibbon2D([[gx0, y], [gx1, y]], 0.28, -0.03, gridCol, 0.0);
    }

    // 2. Greenbelts & urban plazas
    const parks = scene.parks || [];
    for (let p = 0; p < parks.length; p++) {
      const pk = parks[p];
      const fp = pk.footprint;
      const tris = pk.triangles || [];
      const col = pk.color_rgb || [0.15, 0.26, 0.19];
      for (let t = 0; t + 2 < tris.length; t += 3) {
        const a = fp[tris[t]], bPt = fp[tris[t + 1]], c = fp[tris[t + 2]];
        if (!a || !bPt || !c) continue;
        roadBuilder.pushVertex(a[0], a[1], 0.01, 0, 0, 1, col[0], col[1], col[2], 1.0, 0.0, 0.0);
        roadBuilder.pushVertex(bPt[0], bPt[1], 0.01, 0, 0, 1, col[0], col[1], col[2], 1.0, 0.0, 0.0);
        roadBuilder.pushVertex(c[0], c[1], 0.01, 0, 0, 1, col[0], col[1], col[2], 1.0, 0.0, 0.0);
      }
    }

    // 3. Multi-lane asphalt road ribbons + curb shoulders
    const lanes = scene.lanes || [];
    const shoulderCol = [0.22, 0.25, 0.30, 1.0];
    const asphaltCol = [0.16, 0.18, 0.22, 1.0];
    for (let i = 0; i < lanes.length; i++) {
      const ln = lanes[i];
      roadBuilder.pushRibbon2D(ln.shape, ln.width_m + 0.55, 0.03, shoulderCol, 0.0);
      roadBuilder.pushRibbon2D(ln.shape, ln.width_m * 0.98, 0.05, asphaltCol, 0.0);
    }

    // 4. Triangulated junction asphalt polygons
    const junctions = scene.junctions || [];
    const juncCol = [0.15, 0.17, 0.21, 1.0];
    for (let j = 0; j < junctions.length; j++) {
      const jn = junctions[j];
      const poly = jn.polygon || [];
      const tris = jn.triangles || [];
      if (poly.length >= 3 && tris.length >= 3) {
        for (let t = 0; t + 2 < tris.length; t += 3) {
          const p0 = poly[tris[t]], p1 = poly[tris[t + 1]], p2 = poly[tris[t + 2]];
          if (!p0 || !p1 || !p2) continue;
          roadBuilder.pushVertex(p0[0], p0[1], 0.055, 0, 0, 1, juncCol[0], juncCol[1], juncCol[2], 1.0, 0.0, 0.0);
          roadBuilder.pushVertex(p1[0], p1[1], 0.055, 0, 0, 1, juncCol[0], juncCol[1], juncCol[2], 1.0, 0.0, 0.0);
          roadBuilder.pushVertex(p2[0], p2[1], 0.055, 0, 0, 1, juncCol[0], juncCol[1], juncCol[2], 1.0, 0.0, 0.0);
        }
      }
    }

    // 5. Lane dividers, stop bars, zebra crosswalks, and directional turn arrows
    const dividers = scene.lane_dividers || [];
    for (let i = 0; i < dividers.length; i++) {
      const dv = dividers[i];
      if (dv.kind === "yellow_center") {
        roadBuilder.pushRibbon2D(dv.points, 0.18, 0.075, [0.88, 0.70, 0.22, 1.0], 0.0);
      } else {
        roadBuilder.pushDashedRibbon2D(dv.points, 0.13, 2.8, 3.4, 0.075, [0.78, 0.82, 0.88, 1.0]);
      }
    }

    const stopBars = scene.stop_bars || [];
    for (let i = 0; i < stopBars.length; i++) {
      const sb = stopBars[i];
      roadBuilder.pushRibbon2D([sb.p0, sb.p1], 0.42, 0.082, [0.92, 0.94, 0.97, 1.0], 0.0);
    }

    const crosswalks = scene.crosswalks || [];
    for (let i = 0; i < crosswalks.length; i++) {
      const cw = crosswalks[i];
      const yaw = cw.yaw_ccw_rad;
      const cx = cw.xy[0], cy = cw.xy[1];
      const nx = -Math.sin(yaw), ny = Math.cos(yaw);
      const fx = Math.cos(yaw), fy = Math.sin(yaw);
      const nStripes = Math.max(3, Math.floor(cw.width_m / 1.15));
      const span = cw.width_m * 0.86;
      for (let s = 0; s < nStripes; s++) {
        const lat = -0.5 * span + (nStripes > 1 ? (s / (nStripes - 1)) * span : 0.0);
        const sx = cx + nx * lat;
        const sy = cy + ny * lat;
        roadBuilder.pushRibbon2D(
          [
            [sx - fx * 0.95, sy - fy * 0.95],
            [sx + fx * 0.95, sy + fy * 0.95],
          ],
          0.42,
          0.078,
          [0.82, 0.85, 0.90, 1.0],
          0.0
        );
      }
    }

    const arrows = scene.turn_arrows || [];
    const arrowCol = [0.74, 0.79, 0.86, 1.0];
    for (let i = 0; i < arrows.length; i++) {
      const ar = arrows[i];
      const cx = ar.xy[0], cy = ar.xy[1], yaw = ar.yaw_ccw_rad;
      const fx = Math.cos(yaw), fy = Math.sin(yaw);
      const lx = -fy, ly = fx;
      // Stem
      roadBuilder.pushRibbon2D(
        [[cx - fx * 1.3, cy - fy * 1.3], [cx + fx * 0.8, cy + fy * 0.8]],
        0.22,
        0.08,
        arrowCol,
        0.0
      );
      if (ar.dirs.indexOf("S") >= 0) {
        roadBuilder.pushRibbon2D(
          [[cx + fx * 0.5, cy + fy * 0.5], [cx + fx * 1.6, cy + fy * 1.6]],
          0.34,
          0.08,
          arrowCol,
          0.0
        );
      }
      if (ar.dirs.indexOf("L") >= 0) {
        roadBuilder.pushRibbon2D(
          [[cx + fx * 0.4, cy + fy * 0.4], [cx + fx * 0.9 + lx * 0.85, cy + fy * 0.9 + ly * 0.85]],
          0.26,
          0.08,
          arrowCol,
          0.0
        );
      }
      if (ar.dirs.indexOf("R") >= 0) {
        roadBuilder.pushRibbon2D(
          [[cx + fx * 0.4, cy + fy * 0.4], [cx + fx * 0.9 - lx * 0.85, cy + fy * 0.9 - ly * 0.85]],
          0.26,
          0.08,
          arrowCol,
          0.0
        );
      }
    }

    // 6. Static 3D Traffic Signal Poles & Gantries
    const signals = scene.signals || [];
    const poleCol = [0.24, 0.28, 0.34, 1.0];
    const housingCol = [0.12, 0.14, 0.18, 1.0];
    for (let s = 0; s < signals.length; s++) {
      const sig = signals[s];
      const poles = sig.poles || [];
      for (let p = 0; p < poles.length; p++) {
        const pl = poles[p];
        roadBuilder.pushOrientedBox(
          pl.base_xy[0],
          pl.base_xy[1],
          0.0,
          pl.height_m || 5.6,
          pl.yaw_ccw_rad || 0.0,
          0.16,
          0.16,
          poleCol,
          0.0,
          0.85
        );
      }
      const links = sig.links || [];
      for (let l = 0; l < links.length; l++) {
        const lk = links[l];
        const hx = lk.head_xyz[0], hy = lk.head_xyz[1], hz = lk.head_xyz[2];
        roadBuilder.pushOrientedBox(hx, hy, hz - 0.35, hz + 0.35, lk.yaw_ccw_rad, 0.22, 0.32, housingCol, 0.0, 1.0);
      }
    }

    // 7. Extruded 3D Buildings (toggleable layer)
    const buildings = scene.buildings || [];
    for (let bIdx = 0; bIdx < buildings.length; bIdx++) {
      const bldg = buildings[bIdx];
      const fp = bldg.footprint;
      const tris = bldg.triangles || [];
      const h = bldg.height_m || 12.0;
      const col = bldg.color_rgb || [0.76, 0.78, 0.82];
      const roofCol = [col[0] * 0.72, col[1] * 0.74, col[2] * 0.77, 1.0];
      const wallCol = [col[0], col[1], col[2], 1.0];

      // Extruded perimeter walls with continuous facade UV for window mullions
      let perimU = 0.0;
      for (let i = 0; i < fp.length; i++) {
        const p0 = fp[i];
        const p1 = fp[(i + 1) % fp.length];
        const dx = p1[0] - p0[0], dy = p1[1] - p0[1];
        const segLen = Math.hypot(dx, dy);
        if (segLen < 0.05) continue;
        // Outward normal for CCW polygon is (+dy, -dx) / segLen
        const nx = dy / segLen, ny = -dx / segLen;
        buildingBuilder.pushQuad(
          [p0[0], p0[1], 0.0],
          [p1[0], p1[1], 0.0],
          [p1[0], p1[1], h],
          [p0[0], p0[1], h],
          [nx, ny, 0.0],
          wallCol,
          1.0,
          perimU,
          perimU + segLen
        );
        perimU += segLen;
      }

      // Triangulated flat roof
      for (let t = 0; t + 2 < tris.length; t += 3) {
        const a = fp[tris[t]], bPt = fp[tris[t + 1]], c = fp[tris[t + 2]];
        if (!a || !bPt || !c) continue;
        buildingBuilder.pushVertex(a[0], a[1], h, 0, 0, 1, roofCol[0], roofCol[1], roofCol[2], 1.0, 0.0, 0.0);
        buildingBuilder.pushVertex(bPt[0], bPt[1], h, 0, 0, 1, roofCol[0], roofCol[1], roofCol[2], 1.0, 0.0, 0.0);
        buildingBuilder.pushVertex(c[0], c[1], h, 0, 0, 1, roofCol[0], roofCol[1], roofCol[2], 1.0, 0.0, 0.0);
      }
    }

    return {
      roadFloats: roadBuilder.data.subarray(0, roadBuilder.len),
      roadVertexCount: roadBuilder.vertexCount,
      buildingFloats: buildingBuilder.data.subarray(0, buildingBuilder.len),
      buildingVertexCount: buildingBuilder.vertexCount,
    };
  }

  // ===========================================================================
  // 5. Frame Pose Interpolation (Smooth 60-FPS Cubic Hermite Pose Blending)
  // ===========================================================================

  function buildFrameVehicleIndex(trace) {
    const frames = trace.frames || [];
    for (let i = 0; i < frames.length; i++) {
      if (!frames[i]._vehMap) {
        const m = new Map();
        const vs = frames[i].vehicles || [];
        for (let j = 0; j < vs.length; j++) {
          m.set(vs[j].id, vs[j]);
        }
        frames[i]._vehMap = m;
      }
    }
  }

  function interpolateVehicles(trace, cursorFloat) {
    const frames = trace.frames || [];
    if (frames.length === 0) {
      return { frame: null, vehicles: [], vehById: new Map(), simTime: 0.0 };
    }
    const maxIdx = frames.length - 1;
    const clamped = Math.max(0.0, Math.min(maxIdx, cursorFloat));
    const idx0 = Math.floor(clamped);
    const idx1 = Math.min(maxIdx, idx0 + 1);
    const alpha = clamped - idx0;

    const f0 = frames[idx0];
    const f1 = frames[idx1];
    const dt = Math.max(0.05, (f1.t - f0.t) || trace.dt || 0.5);
    const simTime = f0.t + alpha * (f1.t - f0.t);

    const outVehicles = [];
    const vehById = new Map();
    const vs0 = f0.vehicles || [];
    const map1 = f1._vehMap;

    // Cubic Hermite basis functions
    const a2 = alpha * alpha;
    const a3 = a2 * alpha;
    const h00 = 2 * a3 - 3 * a2 + 1;
    const h10 = a3 - 2 * a2 + alpha;
    const h01 = -2 * a3 + 3 * a2;
    const h11 = a3 - a2;

    for (let i = 0; i < vs0.length; i++) {
      const v0 = vs0[i];
      const v1 = idx1 !== idx0 && map1 ? map1.get(v0.id) : null;
      let ix = v0.x, iy = v0.y, iyaw = v0.yaw, iv = v0.v;
      if (v1) {
        const jump = Math.hypot(v1.x - v0.x, v1.y - v0.y);
        if (jump < 35.0) {
          const m0x = Math.cos(v0.yaw) * v0.v * dt;
          const m0y = Math.sin(v0.yaw) * v0.v * dt;
          const m1x = Math.cos(v1.yaw) * v1.v * dt;
          const m1y = Math.sin(v1.yaw) * v1.v * dt;
          ix = h00 * v0.x + h10 * m0x + h01 * v1.x + h11 * m1x;
          iy = h00 * v0.y + h10 * m0y + h01 * v1.y + h11 * m1y;
          iyaw = v0.yaw + wrapAngle(v1.yaw - v0.yaw) * alpha;
          iv = v0.v + (v1.v - v0.v) * alpha;
        }
      }
      const blended = {
        raw: v0,
        id: v0.id,
        vtype: v0.vtype,
        x: ix,
        y: iy,
        yaw: iyaw,
        v: iv,
        a: v0.a,
        L: v0.L || 4.5,
        W: v0.W || 1.85,
        brake: v0.brake,
        turn: v0.turn,
        goal: v0.goal,
        corridor: v0.corridor,
        mode: v0.mode,
        pos_std: v0.pos_std || 0.0,
        entropy: v0.entropy || 0.0,
        obs_age: v0.obs_age || 0.0,
        d_pred: v0.d_pred || 0.0,
        d_post: v0.d_post || 0.0,
        psi: v0.psi || 0.0,
        r_score: v0.r_score || 0.0,
      };
      outVehicles.push(blended);
      vehById.set(v0.id, blended);
    }

    return {
      frame: alpha < 0.5 ? f0 : f1,
      vehicles: outVehicles,
      vehById,
      simTime,
    };
  }

  // ===========================================================================
  // 6. Single WebGL2 Viewport Renderer
  // ===========================================================================

  class CityWebGL2Viewport {
    constructor(canvas, scene, staticBuffers, trace) {
      this.canvas = canvas;
      this.scene = scene;
      this.trace = trace;
      buildFrameVehicleIndex(trace);

      const gl = canvas.getContext("webgl2", {
        antialias: true,
        alpha: false,
        depth: true,
        preserveDrawingBuffer: true,
      });
      if (!gl) {
        throw new Error("WebGL2 is required to run the 3D City & Neural DBN Visualizer.");
      }
      this.gl = gl;

      this.litProg = createProgram(gl, LIT_VS, LIT_FS);
      this.litUniforms = {
        mvp: gl.getUniformLocation(this.litProg, "u_mvp"),
        eyePos: gl.getUniformLocation(this.litProg, "u_eyePos"),
        sceneCenter: gl.getUniformLocation(this.litProg, "u_sceneCenter"),
        fogScale: gl.getUniformLocation(this.litProg, "u_fogScale"),
      };

      this.arcProg = createProgram(gl, ARC_VS, ARC_FS);
      this.arcUniforms = {
        mvp: gl.getUniformLocation(this.arcProg, "u_mvp"),
        animTime: gl.getUniformLocation(this.arcProg, "u_animTime"),
      };

      this.roadVao = this._createLitVao(staticBuffers.roadFloats, gl.STATIC_DRAW);
      this.roadCount = staticBuffers.roadVertexCount;

      this.bldgVao = this._createLitVao(staticBuffers.buildingFloats, gl.STATIC_DRAW);
      this.bldgCount = staticBuffers.buildingVertexCount;

      this.dynLitBuilder = new LitMeshBuilder(131072);
      const dynLit = this._createStreamLitVao();
      this.dynLitVao = dynLit.vao;
      this.dynLitVbo = dynLit.vbo;

      this.dynArcBuilder = new ArcMeshBuilder(131072);
      const dynArc = this._createStreamArcVao();
      this.dynArcVao = dynArc.vao;
      this.dynArcVbo = dynArc.vbo;

      this.lastMvp = mat4Identity();
      this.lastInterpolated = { frame: null, vehicles: [], vehById: new Map(), simTime: 0.0 };
    }

    _createLitVao(floatArray, usage) {
      const gl = this.gl;
      const vao = gl.createVertexArray();
      const vbo = gl.createBuffer();
      gl.bindVertexArray(vao);
      gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
      gl.bufferData(gl.ARRAY_BUFFER, floatArray, usage);
      const stride = 12 * 4;
      gl.enableVertexAttribArray(0);
      gl.vertexAttribPointer(0, 3, gl.FLOAT, false, stride, 0);
      gl.enableVertexAttribArray(1);
      gl.vertexAttribPointer(1, 3, gl.FLOAT, false, stride, 3 * 4);
      gl.enableVertexAttribArray(2);
      gl.vertexAttribPointer(2, 4, gl.FLOAT, false, stride, 6 * 4);
      gl.enableVertexAttribArray(3);
      gl.vertexAttribPointer(3, 2, gl.FLOAT, false, stride, 10 * 4);
      gl.bindVertexArray(null);
      return vao;
    }

    _createStreamLitVao() {
      const gl = this.gl;
      const vao = gl.createVertexArray();
      const vbo = gl.createBuffer();
      gl.bindVertexArray(vao);
      gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
      gl.bufferData(gl.ARRAY_BUFFER, 4 * 12, gl.DYNAMIC_DRAW);
      const stride = 12 * 4;
      gl.enableVertexAttribArray(0);
      gl.vertexAttribPointer(0, 3, gl.FLOAT, false, stride, 0);
      gl.enableVertexAttribArray(1);
      gl.vertexAttribPointer(1, 3, gl.FLOAT, false, stride, 3 * 4);
      gl.enableVertexAttribArray(2);
      gl.vertexAttribPointer(2, 4, gl.FLOAT, false, stride, 6 * 4);
      gl.enableVertexAttribArray(3);
      gl.vertexAttribPointer(3, 2, gl.FLOAT, false, stride, 10 * 4);
      gl.bindVertexArray(null);
      return { vao, vbo };
    }

    _createStreamArcVao() {
      const gl = this.gl;
      const vao = gl.createVertexArray();
      const vbo = gl.createBuffer();
      gl.bindVertexArray(vao);
      gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
      gl.bufferData(gl.ARRAY_BUFFER, 4 * 11, gl.DYNAMIC_DRAW);
      const stride = 11 * 4;
      gl.enableVertexAttribArray(0);
      gl.vertexAttribPointer(0, 3, gl.FLOAT, false, stride, 0);
      gl.enableVertexAttribArray(1);
      gl.vertexAttribPointer(1, 2, gl.FLOAT, false, stride, 3 * 4);
      gl.enableVertexAttribArray(2);
      gl.vertexAttribPointer(2, 4, gl.FLOAT, false, stride, 5 * 4);
      gl.enableVertexAttribArray(3);
      gl.vertexAttribPointer(3, 2, gl.FLOAT, false, stride, 9 * 4);
      gl.bindVertexArray(null);
      return { vao, vbo };
    }

    resizeIfNeeded() {
      const dpr = Math.min(2.0, window.devicePixelRatio || 1.0);
      const cw = Math.max(1, Math.floor(this.canvas.clientWidth * dpr));
      const ch = Math.max(1, Math.floor(this.canvas.clientHeight * dpr));
      if (this.canvas.width !== cw || this.canvas.height !== ch) {
        this.canvas.width = cw;
        this.canvas.height = ch;
      }
      return { width: cw, height: ch, aspect: cw / Math.max(1, ch) };
    }

    buildDynamicFrameMeshes(interp, eyePos, wallClockSec, opts) {
      const lit = this.dynLitBuilder;
      const arc = this.dynArcBuilder;
      lit.clear();
      arc.clear();

      const frame = interp.frame;
      if (!frame) return;

      // 1. Live 3D Traffic Signal Emissive Bulbs
      const tlsStates = frame.tls || {};
      const signals = this.scene.signals || [];
      for (let s = 0; s < signals.length; s++) {
        const sig = signals[s];
        const stateStr = tlsStates[sig.tls_id] || "";
        const links = sig.links || [];
        for (let l = 0; l < links.length; l++) {
          const lk = links[l];
          const ch = stateStr.charAt(lk.link_index) || "G";
          let bulbCol = [0.18, 0.94, 0.48, 1.0];
          if (ch === "r" || ch === "R") {
            bulbCol = [0.98, 0.22, 0.26, 1.0];
          } else if (ch === "y" || ch === "Y") {
            bulbCol = [0.99, 0.78, 0.16, 1.0];
          }
          lit.pushOrientedBox(
            lk.head_xyz[0],
            lk.head_xyz[1],
            lk.head_xyz[2] - 0.18,
            lk.head_xyz[2] + 0.18,
            lk.yaw_ccw_rad,
            0.25,
            0.22,
            bulbCol,
            3.0,
            1.0
          );
        }
      }

      // 2. Sculpted 3D Vehicles + Brake/Turn Lamps + Uncertainty Rings
      const blinkOn = Math.sin(wallClockSec * 14.0) > 0.0;
      const vs = interp.vehicles;
      for (let i = 0; i < vs.length; i++) {
        const v = vs[i];
        let bodyCol;
        if (opts.colorMode === "psi") {
          bodyCol = cascadeRampRgb(v.psi);
        } else if (opts.colorMode === "uncertainty") {
          bodyCol = uncertaintyRampRgb(v.pos_std);
        } else if (opts.colorMode === "goal") {
          bodyCol = hashColorFromId(v.goal);
        } else {
          bodyCol = speedRampRgb(v.v);
        }

        const hl = 0.5 * v.L;
        const hw = 0.5 * v.W;
        const isVan = v.L > 5.4 || (v.vtype && v.vtype.indexOf("van") >= 0);
        const bodyTop = isVan ? 2.15 : 0.92;
        const c = Math.cos(v.yaw), s = Math.sin(v.yaw);

        // Lower body / van cargo box
        lit.pushOrientedBox(v.x, v.y, 0.14, bodyTop, v.yaw, hl, hw, [bodyCol[0], bodyCol[1], bodyCol[2], 1.0], 2.0, 0.96);

        // Passenger cabin greenhouse
        if (!isVan) {
          const cabShift = -0.15 * hl;
          const cabX = v.x + cabShift * c;
          const cabY = v.y + cabShift * s;
          lit.pushOrientedBox(cabX, cabY, 0.90, 1.50, v.yaw, 0.56 * hl, 0.84 * hw, [0.16, 0.22, 0.30, 1.0], 2.0, 0.80);
        } else {
          const cabX = v.x + 0.65 * hl * c;
          const cabY = v.y + 0.65 * hl * s;
          lit.pushOrientedBox(cabX, cabY, 1.05, 1.95, v.yaw, 0.34 * hl, 0.92 * hw, [0.16, 0.22, 0.30, 1.0], 2.0, 0.92);
        }

        // Front headlights
        for (const side of [-0.66, 0.66]) {
          const lx = v.x + (hl + 0.03) * c - side * hw * s;
          const ly = v.y + (hl + 0.03) * s + side * hw * c;
          lit.pushOrientedBox(lx, ly, 0.42, 0.60, v.yaw, 0.08, 0.18, [0.96, 0.97, 1.0, 1.0], 3.0, 1.0);
        }

        // Glowing red brake tail-lights
        const brakeCol = v.brake ? [1.0, 0.14, 0.18, 1.0] : [0.42, 0.08, 0.10, 1.0];
        for (const side of [-0.66, 0.66]) {
          const bx = v.x - (hl + 0.03) * c - side * hw * s;
          const by = v.y - (hl + 0.03) * s + side * hw * c;
          lit.pushOrientedBox(bx, by, 0.46, 0.66, v.yaw, 0.09, 0.22, brakeCol, 3.0, 1.0);
        }

        // Blinking amber turn indicators (turn: +1 = left, -1 = right)
        if (v.turn !== 0 && blinkOn) {
          const sideSign = v.turn > 0 ? 0.86 : -0.86;
          for (const endSign of [-0.96, 0.96]) {
            const tx = v.x + endSign * hl * c - sideSign * hw * s;
            const ty = v.y + endSign * hl * s + sideSign * hw * c;
            lit.pushOrientedBox(tx, ty, 0.48, 0.68, v.yaw, 0.12, 0.14, [1.0, 0.72, 0.10, 1.0], 3.0, 1.0);
          }
        }

        // Selected vehicle gold ring
        if (opts.selectedVehId && v.id === opts.selectedVehId) {
          arc.pushRing2D(v.x, v.y, 0.16, hl + 0.9, hl + 1.65, 24, [1.0, 0.88, 0.26, 0.92]);
        }

        // 3D Kalman Belief Uncertainty Ring
        if (opts.showUncertainty && v.pos_std > 0.08) {
          const rIn = Math.max(1.6, 1.2 + v.pos_std * 1.15);
          const rOut = rIn + 0.55 + Math.min(1.4, v.pos_std * 0.25);
          arc.pushRing2D(v.x, v.y, 0.13, rIn, rOut, 20, [0.32, 0.84, 0.98, Math.min(0.68, 0.22 + 0.14 * v.pos_std)]);
        }
      }

      // 3. Corridor Spillback Heat Overlays
      if (opts.showSpillback && frame.corridor_heat) {
        const ribbons = this.scene.corridor_ribbons || {};
        const entries = Object.entries(frame.corridor_heat);
        for (let e = 0; e < entries.length; e++) {
          const cid = entries[e][0];
          const heat = entries[e][1];
          const intensity = Math.max(heat.spill || 0.0, (heat.tail || 0.0) * 0.9, (heat.occ || 0.0) * 0.75);
          if (intensity < 0.12) continue;
          const shapes = ribbons[cid];
          if (!shapes) continue;
          const hCol = [
            0.98,
            Math.max(0.16, 0.76 - 0.62 * intensity),
            0.14,
            Math.min(0.62, 0.14 + 0.52 * intensity),
          ];
          for (let r = 0; r < shapes.length; r++) {
            const pts = shapes[r];
            for (let k = 0; k < pts.length - 1; k++) {
              const p0 = pts[k], p1 = pts[k + 1];
              const dx = p1[0] - p0[0], dy = p1[1] - p0[1];
              const slen = Math.hypot(dx, dy);
              if (slen < 0.1) continue;
              const nx = (-dy / slen) * 1.65, ny = (dx / slen) * 1.65;
              arc.pushQuadUV(
                [p0[0] - nx, p0[1] - ny, 0.11],
                [p1[0] - nx, p1[1] - ny, 0.11],
                [p1[0] + nx, p1[1] + ny, 0.11],
                [p0[0] + nx, p0[1] + ny, 0.11],
                0.0,
                1.0,
                hCol,
                intensity,
                -1.0
              );
            }
          }
        }
      }

      // 4. V2X Polled Vehicle Vertical Beams
      if (opts.showV2X && frame.v2x_polls) {
        for (let p = 0; p < frame.v2x_polls.length; p++) {
          const pv = interp.vehById.get(frame.v2x_polls[p]);
          if (!pv) continue;
          arc.pushParabolicArc3D(
            pv.x,
            pv.y,
            pv.x + 0.9,
            pv.y + 0.9,
            1.6,
            9.5,
            0.22,
            [0.36, 0.96, 0.88, 0.62],
            0.9,
            0.9,
            eyePos
          );
        }
      }

      // 5. Elevated 3D Parabolic Neural DBN Dependency Arcs
      if (opts.showDBN && frame.dbn_edges) {
        const edges = frame.dbn_edges;
        for (let e = 0; e < edges.length; e++) {
          const ed = edges[e];
          if (ed.omega < opts.minOmega) continue;
          const relIdx = Math.max(0, Math.min(3, ed.rel | 0));
          if (!opts.relMask[relIdx]) continue;
          const srcV = interp.vehById.get(ed.src);
          const dstV = interp.vehById.get(ed.dst);
          if (!srcV || !dstV) continue;

          const span = Math.hypot(dstV.x - srcV.x, dstV.y - srcV.y);
          const apexH = Math.min(26.0, 3.6 + 0.16 * span + 1.8 * relIdx);
          const halfW = 0.22 + 0.56 * ed.omega;
          const rc = REL_COLORS[relIdx];
          const alpha = Math.min(0.95, 0.26 + 0.72 * ed.omega);

          arc.pushParabolicArc3D(
            srcV.x,
            srcV.y,
            dstV.x,
            dstV.y,
            1.65,
            apexH,
            halfW,
            [rc[0], rc[1], rc[2], alpha],
            ed.omega,
            ed.gamma,
            eyePos
          );
        }
      }
    }

    render(cameraState, cursorFloat, wallClockSec, opts) {
      const gl = this.gl;
      const { width, height, aspect } = this.resizeIfNeeded();
      gl.viewport(0, 0, width, height);
      gl.clearColor(0.055, 0.08, 0.125, 1.0);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      gl.enable(gl.DEPTH_TEST);
      gl.depthFunc(gl.LEQUAL);

      const interp = interpolateVehicles(this.trace, cursorFloat);
      this.lastInterpolated = interp;

      let eye = cameraState.eye;
      let target = cameraState.target;
      if (opts.cameraMode === "chase" && opts.selectedVehId) {
        const chased = interp.vehById.get(opts.selectedVehId);
        if (chased) {
          const c = Math.cos(chased.yaw), s = Math.sin(chased.yaw);
          target = [chased.x + c * 8.0, chased.y + s * 8.0, 1.6];
          eye = [chased.x - c * 26.0, chased.y - s * 26.0, 13.5];
        }
      }

      const proj = opts.cameraMode === "ortho"
        ? mat4Ortho(cameraState.distance * 0.52, aspect, -1500.0, 3500.0)
        : mat4Perspective((46.0 * Math.PI) / 180.0, aspect, 1.5, 4500.0);
      const view = mat4LookAtZUp(eye, target, opts.cameraMode === "ortho" ? [0, 1, 0] : [0, 0, 1]);
      const mvp = mat4Multiply(proj, view);
      this.lastMvp = mvp;

      this.buildDynamicFrameMeshes(interp, eye, wallClockSec, opts);

      const b = this.scene.bounds;
      const fogScale = Math.max(260.0, 0.85 * Math.hypot(b.span_x, b.span_y));

      // Pass 1: Opaque Lit Geometry (Roads, Signals, Buildings, 3D Vehicles)
      gl.disable(gl.BLEND);
      gl.depthMask(true);
      gl.useProgram(this.litProg);
      gl.uniformMatrix4fv(this.litUniforms.mvp, false, mvp);
      gl.uniform3f(this.litUniforms.eyePos, eye[0], eye[1], eye[2]);
      gl.uniform3f(this.litUniforms.sceneCenter, b.center_x, b.center_y, 0.0);
      gl.uniform1f(this.litUniforms.fogScale, fogScale);

      gl.bindVertexArray(this.roadVao);
      gl.drawArrays(gl.TRIANGLES, 0, this.roadCount);

      if (opts.showBuildings && this.bldgCount > 0) {
        gl.bindVertexArray(this.bldgVao);
        gl.drawArrays(gl.TRIANGLES, 0, this.bldgCount);
      }

      const dynLitVerts = this.dynLitBuilder.vertexCount;
      if (dynLitVerts > 0) {
        gl.bindVertexArray(this.dynLitVao);
        gl.bindBuffer(gl.ARRAY_BUFFER, this.dynLitVbo);
        gl.bufferData(
          gl.ARRAY_BUFFER,
          this.dynLitBuilder.data.subarray(0, this.dynLitBuilder.len),
          gl.DYNAMIC_DRAW
        );
        gl.drawArrays(gl.TRIANGLES, 0, dynLitVerts);
      }

      // Pass 2: Translucent & Glowing 3D DBN Arcs / Uncertainty Rings / Spillback Overlays
      const dynArcVerts = this.dynArcBuilder.vertexCount;
      if (dynArcVerts > 0) {
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
        gl.depthMask(false);
        gl.useProgram(this.arcProg);
        gl.uniformMatrix4fv(this.arcUniforms.mvp, false, mvp);
        gl.uniform1f(this.arcUniforms.animTime, wallClockSec);

        gl.bindVertexArray(this.dynArcVao);
        gl.bindBuffer(gl.ARRAY_BUFFER, this.dynArcVbo);
        gl.bufferData(
          gl.ARRAY_BUFFER,
          this.dynArcBuilder.data.subarray(0, this.dynArcBuilder.len),
          gl.DYNAMIC_DRAW
        );
        gl.drawArrays(gl.TRIANGLES, 0, dynArcVerts);
        gl.depthMask(true);
        gl.disable(gl.BLEND);
      }

      gl.bindVertexArray(null);
      return interp;
    }

    pickVehicleAtClientXY(clientX, clientY) {
      const rect = this.canvas.getBoundingClientRect();
      const dpr = this.canvas.width / Math.max(1, rect.width);
      const sx = (clientX - rect.left) * dpr;
      const sy = (clientY - rect.top) * dpr;
      const vs = this.lastInterpolated.vehicles || [];

      let bestId = null;
      let bestScore = 34.0 * dpr;
      for (let i = 0; i < vs.length; i++) {
        const v = vs[i];
        const proj = projectWorldToScreen(this.lastMvp, v.x, v.y, 1.0, this.canvas.width, this.canvas.height);
        if (!proj) continue;
        const distPx = Math.hypot(proj[0] - sx, proj[1] - sy);
        if (distPx < bestScore) {
          bestScore = distPx;
          bestId = v.id;
        }
      }
      return bestId;
    }
  }

  // ===========================================================================
  // 7. Interactive Application Controller (Single + Synchronized Split-Screen)
  // ===========================================================================

  function mountCityDBNViewer(payload) {
    const scene = payload.scene;
    const primaryTrace = payload.primary_trace;
    const compareTrace = payload.comparison_trace || null;
    const hasCompare = Boolean(compareTrace && compareTrace.frames && compareTrace.frames.length > 0);

    const staticBuffers = buildStaticCityBuffers(scene);
    const canvasPrimary = document.getElementById("canvas-primary");
    const canvasCompare = document.getElementById("canvas-compare");
    const panePrimary = document.getElementById("pane-primary");
    const paneCompare = document.getElementById("pane-compare");

    const vpPrimary = new CityWebGL2Viewport(canvasPrimary, scene, staticBuffers, primaryTrace);
    const vpCompare = hasCompare
      ? new CityWebGL2Viewport(canvasCompare, scene, staticBuffers, compareTrace)
      : null;

    const b = scene.bounds;
    const initSpan = Math.max(120.0, Math.hypot(b.span_x, b.span_y));
    const cam = {
      targetX: b.center_x,
      targetY: b.center_y,
      yaw: -Math.PI * 0.62,
      pitch: 0.72,
      distance: Math.min(950.0, initSpan * 0.68),
    };

    const opts = {
      playing: true,
      speedMult: 1.0,
      cameraMode: "orbit", // "orbit" | "ortho" | "chase"
      layoutMode: hasCompare ? "split" : "primary",
      colorMode: "speed",
      showBuildings: true,
      showDBN: true,
      showUncertainty: true,
      showSpillback: true,
      showV2X: true,
      minOmega: 0.08,
      relMask: [true, true, true, true],
      selectedVehId: null,
    };

    const totalFrames = Math.max(
      1,
      (primaryTrace.frames || []).length,
      hasCompare ? (compareTrace.frames || []).length : 0
    );
    let cursor = 0.0;
    let lastFrameWallMs = performance.now();

    function computeCameraState() {
      if (opts.cameraMode === "ortho") {
        return {
          target: [cam.targetX, cam.targetY, 0.0],
          eye: [cam.targetX, cam.targetY, Math.max(80.0, cam.distance)],
          distance: cam.distance,
        };
      }
      const horiz = cam.distance * Math.cos(cam.pitch);
      const ez = cam.distance * Math.sin(cam.pitch);
      return {
        target: [cam.targetX, cam.targetY, 0.0],
        eye: [
          cam.targetX + horiz * Math.cos(cam.yaw),
          cam.targetY + horiz * Math.sin(cam.yaw),
          Math.max(6.0, ez),
        ],
        distance: cam.distance,
      };
    }

    function applyLayout() {
      const stage = document.getElementById("viewport-stage");
      if (!hasCompare || opts.layoutMode === "primary") {
        stage.dataset.layout = "single";
        panePrimary.style.display = "flex";
        paneCompare.style.display = "none";
      } else if (opts.layoutMode === "compare") {
        stage.dataset.layout = "single";
        panePrimary.style.display = "none";
        paneCompare.style.display = "flex";
      } else {
        stage.dataset.layout = "split";
        panePrimary.style.display = "flex";
        paneCompare.style.display = "flex";
      }
    }

    // Orbit / Pan / Zoom Pointer Bindings (shared across both viewports)
    function bindViewportInteractions(canvas, viewport) {
      let dragging = false;
      let dragButton = 0;
      let startX = 0, startY = 0;
      let movedPx = 0;

      canvas.addEventListener("contextmenu", (ev) => ev.preventDefault());
      canvas.addEventListener("mousedown", (ev) => {
        dragging = true;
        dragButton = ev.button;
        startX = ev.clientX;
        startY = ev.clientY;
        movedPx = 0;
      });

      window.addEventListener("mousemove", (ev) => {
        if (!dragging) return;
        const dx = ev.clientX - startX;
        const dy = ev.clientY - startY;
        startX = ev.clientX;
        startY = ev.clientY;
        movedPx += Math.hypot(dx, dy);

        if (dragButton === 2 || ev.shiftKey || opts.cameraMode === "ortho") {
          const panScale = cam.distance * 0.0016;
          const rx = -Math.sin(cam.yaw), ry = Math.cos(cam.yaw);
          const fx = -Math.cos(cam.yaw), fy = -Math.sin(cam.yaw);
          if (opts.cameraMode === "ortho") {
            cam.targetX -= dx * panScale;
            cam.targetY += dy * panScale;
          } else {
            cam.targetX -= (rx * dx - fx * dy) * panScale;
            cam.targetY -= (ry * dx - fy * dy) * panScale;
          }
        } else {
          cam.yaw -= dx * 0.0065;
          cam.pitch = Math.max(0.14, Math.min(1.48, cam.pitch + dy * 0.0055));
        }
      });

      window.addEventListener("mouseup", (ev) => {
        if (!dragging) return;
        dragging = false;
        if (movedPx < 5.0 && ev.target === canvas) {
          const picked = viewport.pickVehicleAtClientXY(ev.clientX, ev.clientY);
          opts.selectedVehId = picked;
          if (!picked && opts.cameraMode === "chase") {
            opts.cameraMode = "orbit";
            syncCameraButtons();
          }
        }
      });

      canvas.addEventListener(
        "wheel",
        (ev) => {
          ev.preventDefault();
          const factor = Math.exp(ev.deltaY * 0.0012);
          cam.distance = Math.max(24.0, Math.min(2600.0, cam.distance * factor));
        },
        { passive: false }
      );
    }

    bindViewportInteractions(canvasPrimary, vpPrimary);
    if (vpCompare) {
      bindViewportInteractions(canvasCompare, vpCompare);
    }

    // Wire UI Controls
    const btnPlay = document.getElementById("btn-play");
    const sliderScrub = document.getElementById("slider-scrub");
    const lblClock = document.getElementById("lbl-clock");
    const selectSpeed = document.getElementById("select-speed");
    const selectColor = document.getElementById("select-color");
    const sliderOmega = document.getElementById("slider-omega");
    const lblOmega = document.getElementById("lbl-omega");
    const btnResetCam = document.getElementById("btn-reset-cam");
    const btnExportPng = document.getElementById("btn-export-png");
    const sparkCanvas = document.getElementById("sparkline-canvas");

    sliderScrub.max = String(Math.max(0, totalFrames - 1));
    document.getElementById("badge-network").textContent =
      scene.network_id + " (" + (scene.buildings || []).length + " 3D bldgs)";
    document.getElementById("title-primary").textContent =
      (primaryTrace.policy_name || "neural_dbn") + "  •  seed " + primaryTrace.seed;

    const layoutWrap = document.getElementById("layout-group");
    if (hasCompare) {
      document.getElementById("title-compare").textContent =
        (compareTrace.policy_name || "baseline") + "  •  seed " + compareTrace.seed;
      layoutWrap.style.display = "flex";
    } else {
      layoutWrap.style.display = "none";
    }

    btnPlay.addEventListener("click", () => {
      opts.playing = !opts.playing;
      btnPlay.textContent = opts.playing ? "Pause" : "Play";
    });

    sliderScrub.addEventListener("input", () => {
      cursor = parseFloat(sliderScrub.value) || 0.0;
    });

    selectSpeed.addEventListener("change", () => {
      opts.speedMult = parseFloat(selectSpeed.value) || 1.0;
    });

    selectColor.addEventListener("change", () => {
      opts.colorMode = selectColor.value;
    });

    sliderOmega.addEventListener("input", () => {
      opts.minOmega = parseFloat(sliderOmega.value) || 0.05;
      lblOmega.textContent = opts.minOmega.toFixed(2);
    });

    function syncCameraButtons() {
      document.querySelectorAll("[data-cam]").forEach((el) => {
        el.classList.toggle("active", el.dataset.cam === opts.cameraMode);
      });
    }

    document.querySelectorAll("[data-cam]").forEach((el) => {
      el.addEventListener("click", () => {
        const requested = el.dataset.cam;
        if (requested === "chase" && !opts.selectedVehId) {
          const vs = vpPrimary.lastInterpolated.vehicles || [];
          if (vs.length > 0) {
            let best = vs[0];
            for (let i = 1; i < vs.length; i++) {
              if (vs[i].psi > best.psi) best = vs[i];
            }
            opts.selectedVehId = best.id;
          }
        }
        opts.cameraMode = requested;
        syncCameraButtons();
      });
    });

    document.querySelectorAll("[data-layout]").forEach((el) => {
      el.addEventListener("click", () => {
        opts.layoutMode = el.dataset.layout;
        document.querySelectorAll("[data-layout]").forEach((bEl) => {
          bEl.classList.toggle("active", bEl.dataset.layout === opts.layoutMode);
        });
        applyLayout();
      });
    });

    const bindToggle = (id, key) => {
      const el = document.getElementById(id);
      if (!el) return;
      el.addEventListener("change", () => {
        opts[key] = el.checked;
      });
    };
    bindToggle("chk-buildings", "showBuildings");
    bindToggle("chk-dbn", "showDBN");
    bindToggle("chk-uncertainty", "showUncertainty");
    bindToggle("chk-spillback", "showSpillback");
    bindToggle("chk-v2x", "showV2X");

    for (let r = 0; r < 4; r++) {
      const el = document.getElementById("chk-rel-" + r);
      if (el) {
        el.addEventListener("change", () => {
          opts.relMask[r] = el.checked;
        });
      }
    }

    btnResetCam.addEventListener("click", () => {
      cam.targetX = b.center_x;
      cam.targetY = b.center_y;
      cam.yaw = -Math.PI * 0.62;
      cam.pitch = 0.72;
      cam.distance = Math.min(950.0, initSpan * 0.68);
      opts.cameraMode = "orbit";
      syncCameraButtons();
    });

    btnExportPng.addEventListener("click", () => {
      const exportCanvas = document.createElement("canvas");
      const showBoth = hasCompare && opts.layoutMode === "split";
      const srcLeft = opts.layoutMode === "compare" && vpCompare ? canvasCompare : canvasPrimary;
      const w1 = srcLeft.width;
      const h1 = srcLeft.height;
      const w2 = showBoth && vpCompare ? canvasCompare.width : 0;
      exportCanvas.width = w1 + (showBoth ? w2 + 8 : 0);
      exportCanvas.height = h1 + 56;
      const ctx = exportCanvas.getContext("2d");
      ctx.fillStyle = "#0b111c";
      ctx.fillRect(0, 0, exportCanvas.width, exportCanvas.height);
      ctx.drawImage(srcLeft, 0, 56);
      if (showBoth && vpCompare) {
        ctx.drawImage(canvasCompare, w1 + 8, 56);
      }
      ctx.fillStyle = "#e2e8f0";
      ctx.font = "600 22px system-ui, -apple-system, sans-serif";
      ctx.fillText(
        "City Swarm Neural DBN 3D Replay  |  " +
          scene.network_id +
          "  |  t = " +
          vpPrimary.lastInterpolated.simTime.toFixed(1) +
          "s",
        20,
        36
      );
      const link = document.createElement("a");
      link.download =
        "city_dbn_3d_" +
        scene.network_id +
        "_t" +
        Math.round(vpPrimary.lastInterpolated.simTime) +
        ".png";
      link.href = exportCanvas.toDataURL("image/png");
      link.click();
    });

    function updateKPIsAndInspector(interpP, interpC) {
      const fP = interpP.frame;
      if (fP && fP.kpi) {
        const k = fP.kpi;
        document.getElementById("kpi-p-veh").textContent = k.active_veh + " (" + k.halted_veh + " halted)";
        document.getElementById("kpi-p-speed").textContent = k.mean_speed_mps.toFixed(1) + " m/s";
        document.getElementById("kpi-p-edges").textContent = String(k.active_edges);
        document.getElementById("kpi-p-spill").textContent = String(k.spillback_corridors);
        document.getElementById("kpi-p-lat").textContent = k.dbn_ms.toFixed(2) + " ms";
      }

      if (hasCompare && interpC && interpC.frame && interpC.frame.kpi) {
        const kC = interpC.frame.kpi;
        document.getElementById("kpi-c-veh").textContent = kC.active_veh + " (" + kC.halted_veh + " halted)";
        document.getElementById("kpi-c-speed").textContent = kC.mean_speed_mps.toFixed(1) + " m/s";
        document.getElementById("kpi-c-spill").textContent = String(kC.spillback_corridors);
        if (fP && fP.kpi) {
          const dv = fP.kpi.mean_speed_mps - kC.mean_speed_mps;
          const dh = fP.kpi.halted_veh - kC.halted_veh;
          const badge = document.getElementById("badge-delta");
          badge.style.display = "inline-flex";
          badge.textContent =
            "Δ Speed: " +
            (dv >= 0 ? "+" : "") +
            dv.toFixed(2) +
            " m/s  |  Δ Halted: " +
            (dh > 0 ? "+" : "") +
            dh;
        }
      }

      // Vehicle Belief & DBN Inspector Panel
      const inspBox = document.getElementById("inspector-body");
      if (!opts.selectedVehId) {
        inspBox.innerHTML =
          '<div class="muted">Click any 3D vehicle (or switch to Chase Cam) to inspect its Kalman belief, turn entropy, and live DBN causal arcs.</div>';
        return;
      }
      const v =
        interpP.vehById.get(opts.selectedVehId) ||
        (interpC ? interpC.vehById.get(opts.selectedVehId) : null);
      if (!v) {
        inspBox.innerHTML =
          '<div class="muted">Vehicle <b>' + opts.selectedVehId + "</b> has exited the road network.</div>";
        return;
      }

      const edges = (fP && fP.dbn_edges) || [];
      const inEdges = edges.filter((e) => e.dst === v.id).slice(0, 4);
      const outEdges = edges.filter((e) => e.src === v.id).slice(0, 4);
      const fmtEdges = (arr, dir) =>
        arr.length === 0
          ? '<span class="muted">none</span>'
          : arr
              .map(
                (e) =>
                  '<div class="edge-pill">' +
                  (dir === "in" ? e.src + " → " : "→ " + e.dst + " ") +
                  "<b>ω=" +
                  e.omega.toFixed(2) +
                  "</b> γ=" +
                  e.gamma.toFixed(2) +
                  " <small>(" +
                  REL_NAMES[e.rel || 0] +
                  ")</small></div>"
              )
              .join("");

      inspBox.innerHTML =
        '<div class="insp-grid">' +
        "<div><span>Vehicle</span><b>" + v.id + " (" + v.vtype + ")</b></div>" +
        "<div><span>Action Mode</span><b>" + (MODE_NAMES[String(v.mode)] || "CRUISE") + "</b></div>" +
        "<div><span>Speed / Accel</span><b>" + v.v.toFixed(2) + " m/s / " + v.a.toFixed(2) + "</b></div>" +
        "<div><span>Cascade ψ</span><b>" + v.psi.toFixed(3) + "</b></div>" +
        "<div><span>Delay Post P(D=1)</span><b>" + v.d_post.toFixed(3) + "</b></div>" +
        "<div><span>Belief σ_s / H(π)</span><b>" + v.pos_std.toFixed(2) + "m / " + v.entropy.toFixed(2) + "</b></div>" +
        "<div><span>Corridor → Goal</span><b>" + v.corridor + " → " + v.goal + "</b></div>" +
        "<div><span>Precedence r_i</span><b>" + v.r_score.toFixed(2) + "</b></div>" +
        "</div>" +
        '<div class="insp-sub">Parents in G_t: ' + fmtEdges(inEdges, "in") + "</div>" +
        '<div class="insp-sub">Children in G_t: ' + fmtEdges(outEdges, "out") + "</div>";
    }

    function drawSparkline() {
      if (!sparkCanvas) return;
      const ctx = sparkCanvas.getContext("2d");
      const w = (sparkCanvas.width = sparkCanvas.clientWidth || 300);
      const h = (sparkCanvas.height = sparkCanvas.clientHeight || 36);
      ctx.clearRect(0, 0, w, h);

      const tl = primaryTrace.timeline;
      if (!tl || !tl.t || tl.t.length < 2) return;
      const n = tl.t.length;
      const maxHalt = Math.max(5, ...tl.halted_veh, ...(hasCompare && compareTrace.timeline ? compareTrace.timeline.halted_veh : [1]));

      const plotLine = (series, color) => {
        if (!series || series.length < 2) return;
        ctx.beginPath();
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.6;
        for (let i = 0; i < series.length; i++) {
          const px = (i / (series.length - 1)) * w;
          const py = h - 4 - (series[i] / maxHalt) * (h - 8);
          if (i === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        }
        ctx.stroke();
      };

      if (hasCompare && compareTrace.timeline) {
        plotLine(compareTrace.timeline.halted_veh, "rgba(251, 113, 133, 0.85)");
      }
      plotLine(tl.halted_veh, "rgba(56, 189, 248, 0.95)");

      const playX = (Math.min(n - 1, cursor) / Math.max(1, n - 1)) * w;
      ctx.fillStyle = "#f8fafc";
      ctx.fillRect(playX - 1, 0, 2, h);
    }

    applyLayout();
    syncCameraButtons();

    function tick(nowMs) {
      const elapsedS = Math.min(0.1, (nowMs - lastFrameWallMs) / 1000.0);
      lastFrameWallMs = nowMs;

      const dtSim = primaryTrace.dt || 0.5;
      if (opts.playing && totalFrames > 1) {
        cursor += (elapsedS * opts.speedMult) / dtSim;
        if (cursor >= totalFrames - 1) {
          cursor = 0.0;
        }
        sliderScrub.value = cursor.toFixed(2);
      }

      const camState = computeCameraState();
      const wallSec = nowMs * 0.001;

      let interpP = vpPrimary.lastInterpolated;
      let interpC = vpCompare ? vpCompare.lastInterpolated : null;

      if (panePrimary.style.display !== "none") {
        interpP = vpPrimary.render(camState, cursor, wallSec, opts);
      }
      if (vpCompare && paneCompare.style.display !== "none") {
        const clampedCompare = Math.min(Math.max(0, (compareTrace.frames || []).length - 1), cursor);
        interpC = vpCompare.render(camState, clampedCompare, wallSec, opts);
      }

      const simT = interpP.simTime || 0.0;
      lblClock.textContent = "t = " + simT.toFixed(1) + " s";
      updateKPIsAndInspector(interpP, interpC);
      drawSparkline();

      requestAnimationFrame(tick);
    }

    requestAnimationFrame(tick);
  }

  if (typeof window !== "undefined") {
    window.mountCityDBNViewer = mountCityDBNViewer;
  }
})();
