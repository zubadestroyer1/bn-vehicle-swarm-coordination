/*
 * Cinema-Grade Offline WebGL2 3D City & Neural Dynamic Bayesian Network Engine
 * Pure native WebGL2 + GLSL ES 3.00 — zero external dependencies, zero network calls.
 */
(function () {
  "use strict";

  // Relation palette matching #arcs-legend: Corridor (Sky/Cyan), Conflict (Rose), Spillback (Amber), Wave (Emerald)
  const REL_COLORS = [
    [0.14, 0.82, 0.96], // 0: SAME_CORRIDOR
    [0.99, 0.38, 0.52], // 1: SAME_JUNCTION_CONFLICT
    [0.98, 0.72, 0.15], // 2: DOWNSTREAM_SPILLBACK
    [0.16, 0.86, 0.58], // 3: UPSTREAM_FEEDER (Green Wave)
  ];

  const MODE_META = {
    "-1": { label: "Actuated", slug: "signal" },
    "0": { label: "Go-Wave", slug: "go-wave" },
    "1": { label: "Cruise", slug: "cruise" },
    "2": { label: "Eco-Glide", slug: "eco-glide" },
    "3": { label: "Yield", slug: "yield" },
  };

  // ===========================================================================
  // 1. Async Scheduling, Gzip Base64 Decoding & Flexible Payload Unpacking
  // ===========================================================================

  async function yieldToMain() {
    if (typeof window !== "undefined" && "scheduler" in window && window.scheduler && typeof window.scheduler.yield === "function") {
      return await window.scheduler.yield();
    }
    return new Promise((resolve) => setTimeout(resolve, 0));
  }

  function base64ToBytes(b64) {
    const clean = b64.replace(/\s+/g, "");
    if (typeof Uint8Array.fromBase64 === "function") {
      return Uint8Array.fromBase64(clean);
    }
    const bin = atob(clean);
    const len = bin.length;
    const out = new Uint8Array(len);
    for (let i = 0; i < len; i++) {
      out[i] = bin.charCodeAt(i);
    }
    return out;
  }

  function unpackSoATrace(tr) {
    if (!tr || typeof tr !== "object") return null;
    if (Array.isArray(tr.frames) && tr.frames.length > 0) return tr;
    if (tr.encoding !== "soa_v2" || !Array.isArray(tr.packed_frames)) return tr;

    const strings = tr.strings || [];
    const vehMeta = tr.veh_meta || [];
    const tlsKeys = tr.tls_keys || [];
    const pFrames = tr.packed_frames;
    const frames = new Array(pFrames.length);

    for (let f = 0; f < pFrames.length; f++) {
      const [t, vCols, eCols, hCols, tlsVals, pollV, kArr] = pFrames[f];
      const vehicles = [];
      if (vCols && vCols.length > 0) {
        const [vIdx, xCm, yCm, yawMr, vCms, aCms, corrS, sBits, dPm, psiPm, stdCm] = vCols;
        const hasA = aCms.length > 0;
        const hasD = dPm.length > 0;
        const hasPsi = psiPm.length > 0;
        const hasStd = stdCm.length > 0;
        for (let i = 0, n = vIdx.length; i < n; i++) {
          const vm = vehMeta[vIdx[i]];
          const sb = sBits[i];
          const dp = hasD ? dPm[i] * 0.001 : 0.0;
          vehicles.push({
            id: vm[0],
            vtype: strings[vm[1]],
            x: xCm[i] * 0.01,
            y: yCm[i] * 0.01,
            yaw: yawMr[i] * 0.001,
            v: vCms[i] * 0.01,
            a: hasA ? aCms[i] * 0.01 : 0.0,
            L: vm[2],
            W: vm[3],
            brake: Boolean(sb & 1),
            turn: ((sb >> 1) & 7) - 1,
            goal: strings[vm[4]],
            corridor: strings[corrS[i]],
            mode: ((sb >> 4) & 15) - 1,
            pos_std: hasStd ? stdCm[i] * 0.01 : 0.0,
            entropy: 0.0,
            obs_age: 0.0,
            d_pred: dp,
            d_post: dp,
            psi: hasPsi ? psiPm[i] * 0.001 : 0.0,
            r_score: 0.0,
          });
        }
      }
      const dbn_edges = [];
      if (eCols && eCols.length > 0) {
        const [eSrc, eDst, eRel, eOm, eKa, eGa, ePc] = eCols;
        const hasK = eKa.length > 0;
        const hasG = eGa.length > 0;
        const hasP = ePc.length > 0;
        for (let i = 0, n = eSrc.length; i < n; i++) {
          dbn_edges.push({
            src: vehMeta[eSrc[i]][0],
            dst: vehMeta[eDst[i]][0],
            rel: eRel[i],
            omega: eOm[i] * 0.001,
            kappa: hasK ? eKa[i] * 0.001 : 0.0,
            gamma: hasG ? eGa[i] * 0.001 : 0.0,
            p_conf: hasP ? ePc[i] * 0.001 : 0.0,
          });
        }
      }
      const corridor_heat = {};
      if (hCols && hCols.length > 0) {
        const [hCid, hOcc, hTail, hSpill] = hCols;
        for (let i = 0, n = hCid.length; i < n; i++) {
          corridor_heat[strings[hCid[i]]] = {
            occ: hOcc[i] * 0.001,
            tail: hTail[i] * 0.001,
            spill: hSpill[i] * 0.001,
          };
        }
      }
      const tls = {};
      for (let i = 0; i < tlsKeys.length && i < tlsVals.length; i++) {
        tls[tlsKeys[i]] = tlsVals[i];
      }
      const v2x_polls = new Array(pollV ? pollV.length : 0);
      for (let i = 0; i < v2x_polls.length; i++) {
        v2x_polls[i] = vehMeta[pollV[i]][0];
      }
      frames[f] = {
        t,
        vehicles,
        dbn_edges,
        corridor_heat,
        tls,
        v2x_polls,
        kpi: {
          active_veh: kArr[0],
          halted_veh: kArr[1],
          mean_speed_mps: kArr[2],
          active_edges: kArr[3],
          mean_psi: kArr[4],
          spillback_corridors: kArr[5],
          cum_collisions: kArr[6],
          cum_emerg_brakes: kArr[7],
          cum_reroutes: kArr[8],
          dbn_ms: kArr[9],
        },
      };
    }
    return Object.assign({}, tr, { frames, frame_count: frames.length });
  }

  function hasTraceFrames(tr) {
    if (!tr || typeof tr !== "object") return false;
    if (Array.isArray(tr.frames) && tr.frames.length > 0) return true;
    if (tr.encoding === "soa_v2" && Array.isArray(tr.packed_frames) && tr.packed_frames.length > 0) return true;
    return false;
  }

  function normalizePayload(rawPayload, metaObj) {
    if (!rawPayload || typeof rawPayload !== "object") {
      throw new Error("Decoded 3D replay payload is empty or invalid.");
    }
    const scene = rawPayload.scene;
    const primaryTrace = rawPayload.primary_trace || rawPayload.primary;
    const compareTrace = rawPayload.comparison_trace || rawPayload.compare || null;

    if (!scene || !hasTraceFrames(primaryTrace)) {
      throw new Error("Replay payload is missing valid scene geometry or primary_trace frames.");
    }
    return {
      scene,
      primary_trace: primaryTrace,
      comparison_trace: hasTraceFrames(compareTrace) ? compareTrace : null,
      meta: metaObj || rawPayload.meta || null,
    };
  }

  async function decodeSwarmPayload(setStatus) {
    return decodeSwarmPayloadFromDOM(setStatus);
  }

  async function decodeSwarmPayloadFromDOM(setStatus) {
    let metaObj = null;
    const metaEl = document.getElementById("swarm-payload-meta");
    if (metaEl) {
      const mTxt = (metaEl.textContent || "").trim();
      if (mTxt && mTxt.indexOf("{{") !== 0) {
        metaObj = JSON.parse(mTxt);
      }
    }

    const gzEl = document.getElementById("swarm-payload-gz");
    if (gzEl) {
      const rawB64 = (gzEl.textContent || "").trim();
      if (rawB64 && rawB64.indexOf("{{") !== 0) {
        if (setStatus) setStatus("Decompressing 3D replay stream…");
        await yieldToMain();
        if (typeof DecompressionStream !== "function") {
          throw new Error("This browser does not support native DecompressionStream('gzip').");
        }
        const gzBytes = base64ToBytes(rawB64);
        const ds = new DecompressionStream("gzip");
        const decompressedStream = new Blob([gzBytes]).stream().pipeThrough(ds);
        const jsonText = await new Response(decompressedStream).text();
        if (setStatus) setStatus("Parsing urban geometry & vehicle trajectories…");
        await yieldToMain();
        const rawObj = JSON.parse(jsonText);
        return normalizePayload(rawObj, metaObj);
      }
    }

    const jsonEl = document.getElementById("swarm-payload");
    if (jsonEl) {
      const rawJson = (jsonEl.textContent || "").trim();
      if (rawJson && rawJson.indexOf("{{") !== 0) {
        if (setStatus) setStatus("Parsing 3D replay payload…");
        await yieldToMain();
        const rawObj = JSON.parse(rawJson);
        return normalizePayload(rawObj, metaObj);
      }
    }

    throw new Error("No embedded 3D replay payload found in document (#swarm-payload-gz or #swarm-payload).");
  }

  // ===========================================================================
  // 2. Linear Algebra & Camera Ray Helpers (Right-Handed Z-Up World)
  // ===========================================================================

  function mat4Identity() {
    const m = new Float32Array(16);
    m[0] = 1; m[5] = 1; m[10] = 1; m[15] = 1;
    return m;
  }

  function mat4Multiply(out, a, b) {
    for (let c = 0; c < 4; c++) {
      const b0 = b[c * 4 + 0], b1 = b[c * 4 + 1], b2 = b[c * 4 + 2], b3 = b[c * 4 + 3];
      out[c * 4 + 0] = a[0] * b0 + a[4] * b1 + a[8] * b2 + a[12] * b3;
      out[c * 4 + 1] = a[1] * b0 + a[5] * b1 + a[9] * b2 + a[13] * b3;
      out[c * 4 + 2] = a[2] * b0 + a[6] * b1 + a[10] * b2 + a[14] * b3;
      out[c * 4 + 3] = a[3] * b0 + a[7] * b1 + a[11] * b2 + a[15] * b3;
    }
    return out;
  }

  function mat4Perspective(out, fovYRad, aspect, near, far) {
    out.fill(0);
    const f = 1.0 / Math.tan(0.5 * fovYRad);
    const nf = 1.0 / (near - far);
    out[0] = f / Math.max(0.1, aspect);
    out[5] = f;
    out[10] = (far + near) * nf;
    out[11] = -1.0;
    out[14] = 2.0 * far * near * nf;
    return out;
  }

  function mat4Ortho(out, halfHeight, aspect, near, far) {
    out.fill(0);
    const top = Math.max(8.0, halfHeight);
    const right = top * Math.max(0.1, aspect);
    const nf = 1.0 / (near - far);
    out[0] = 1.0 / right;
    out[5] = 1.0 / top;
    out[10] = 2.0 * nf;
    out[14] = (far + near) * nf;
    out[15] = 1.0;
    return out;
  }

  function mat4LookAtZUp(out, eye, target, upHintX, upHintY, upHintZ) {
    let fx = target[0] - eye[0];
    let fy = target[1] - eye[1];
    let fz = target[2] - eye[2];
    const flen = Math.hypot(fx, fy, fz) || 1.0;
    fx /= flen; fy /= flen; fz /= flen;

    let ux = upHintX, uy = upHintY, uz = upHintZ;
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

    out[0] = sx; out[4] = sy; out[8] = sz; out[12] = -(sx * eye[0] + sy * eye[1] + sz * eye[2]);
    out[1] = vx; out[5] = vy; out[9] = vz; out[13] = -(vx * eye[0] + vy * eye[1] + vz * eye[2]);
    out[2] = -fx; out[6] = -fy; out[10] = -fz; out[14] = fx * eye[0] + fy * eye[1] + fz * eye[2];
    out[3] = 0; out[7] = 0; out[11] = 0; out[15] = 1;
    return out;
  }

  function wrapAngle(rad) {
    return Math.atan2(Math.sin(rad), Math.cos(rad));
  }

  function intersectCursorGroundZ0(ndcX, ndcY, eye, target, fovYRad, aspect, isOrtho, orthoHalfH) {
    let fx = target[0] - eye[0];
    let fy = target[1] - eye[1];
    let fz = target[2] - eye[2];
    const flen = Math.hypot(fx, fy, fz) || 1.0;
    fx /= flen; fy /= flen; fz /= flen;

    let ux = 0.0, uy = 0.0, uz = isOrtho ? 0.0 : 1.0;
    if (isOrtho) { uy = 1.0; }
    if (Math.abs(fx * ux + fy * uy + fz * uz) > 0.995) {
      ux = 0.0; uy = 1.0; uz = 0.0;
    }
    let rx = fy * uz - fz * uy;
    let ry = fz * ux - fx * uz;
    let rz = fx * uy - fy * ux;
    const rlen = Math.hypot(rx, ry, rz) || 1.0;
    rx /= rlen; ry /= rlen; rz /= rlen;

    const vx = ry * fz - rz * fy;
    const vy = rz * fx - rx * fz;
    const vz = rx * fy - ry * fx;

    if (isOrtho) {
      const worldX = target[0] + ndcX * orthoHalfH * aspect * rx + ndcY * orthoHalfH * vx;
      const worldY = target[1] + ndcX * orthoHalfH * aspect * ry + ndcY * orthoHalfH * vy;
      return [worldX, worldY];
    }

    const tanHalf = Math.tan(0.5 * fovYRad);
    const dx = fx + ndcX * aspect * tanHalf * rx + ndcY * tanHalf * vx;
    const dy = fy + ndcX * aspect * tanHalf * ry + ndcY * tanHalf * vy;
    const dz = fz + ndcX * aspect * tanHalf * rz + ndcY * tanHalf * vz;
    if (dz >= -0.02) return null;
    const t = -eye[2] / dz;
    if (t <= 0.0 || t > 6000.0) return null;
    return [eye[0] + dx * t, eye[1] + dy * t];
  }

  function computeSmartFocusBounds(scene) {
    const fb = scene.focus_bounds || (scene.bounds && scene.bounds.focus_bounds);
    if (fb && Number.isFinite(fb.center_x) && Number.isFinite(fb.center_y)) {
      const sx = Math.max(90.0, fb.span_x || (fb.max_x - fb.min_x) || 260.0);
      const sy = Math.max(90.0, fb.span_y || (fb.max_y - fb.min_y) || 260.0);
      return { centerX: fb.center_x, centerY: fb.center_y, spanX: sx, spanY: sy };
    }

    // Extract signalized intersection coordinates first (best urban core proxy)
    const xs = [];
    const ys = [];
    const signals = scene.signals || [];
    for (let s = 0; s < signals.length; s++) {
      const links = signals[s].links || [];
      if (links.length > 0) {
        let sx = 0, sy = 0;
        for (let l = 0; l < links.length; l++) {
          sx += links[l].head_xyz[0];
          sy += links[l].head_xyz[1];
        }
        xs.push(sx / links.length);
        ys.push(sy / links.length);
      }
    }

    if (xs.length < 3) {
      const junctions = scene.junctions || [];
      for (let j = 0; j < junctions.length; j++) {
        const jn = junctions[j];
        if ((jn.polygon && jn.polygon.length >= 3) || junctions.length <= 6) {
          xs.push(jn.coord_xy[0]);
          ys.push(jn.coord_xy[1]);
        }
      }
    }

    if (xs.length >= 2) {
      xs.sort((a, b) => a - b);
      ys.sort((a, b) => a - b);
      const q = (arr, p) => {
        const idx = Math.max(0, Math.min(arr.length - 1, p * (arr.length - 1)));
        const lo = Math.floor(idx);
        const hi = Math.min(arr.length - 1, lo + 1);
        return arr[lo] + (arr[hi] - arr[lo]) * (idx - lo);
      };
      const x05 = q(xs, 0.06), x95 = q(xs, 0.94);
      const y05 = q(ys, 0.06), y95 = q(ys, 0.94);
      const xMed = q(xs, 0.50), yMed = q(ys, 0.50);
      const midX = 0.55 * (0.5 * (x05 + x95)) + 0.45 * xMed;
      const midY = 0.55 * (0.5 * (y05 + y95)) + 0.45 * yMed;
      return {
        centerX: midX,
        centerY: midY,
        spanX: Math.max(120.0, x95 - x05 + 70.0),
        spanY: Math.max(120.0, y95 - y05 + 70.0),
      };
    }

    const b = scene.bounds;
    return {
      centerX: b.center_x,
      centerY: b.center_y,
      spanX: Math.max(120.0, b.span_x),
      spanY: Math.max(120.0, b.span_y),
    };
  }

  // ===========================================================================
  // 3. Vehicle Palette: Cyan-Teal (#22d3ee) -> Soft Gold (#fbbf24) -> Coral (#f43f5e)
  // ===========================================================================

  function writeVehicleColorRGB(outArr, offset, vMps, dPost, modeInt) {
    const speedStress = 1.0 - Math.max(0.0, Math.min(1.0, (vMps - 0.7) / 9.8));
    const postStress = Math.max(0.0, Math.min(1.0, dPost * 1.15));
    let s = Math.max(speedStress * 0.88, postStress * 0.86);

    if (modeInt === 0 && vMps > 1.8) {
      // GO_WAVE: crisp cyan-teal bias
      s = Math.min(s * 0.55, 0.28);
    } else if (modeInt === 2 && vMps > 1.2) {
      // ECO_GLIDE: warm soft-gold glide
      s = 0.44 + 0.22 * s;
    } else if (modeInt === 3) {
      // YIELD: elevated amber-coral alert
      s = Math.max(s, 0.68);
    }

    s = Math.max(0.0, Math.min(1.0, s));
    let r, g, b;
    if (s <= 0.5) {
      const t = s * 2.0;
      // #22d3ee (0.133, 0.827, 0.933) -> #fbbf24 (0.984, 0.749, 0.141)
      r = 0.133 + (0.984 - 0.133) * t;
      g = 0.827 + (0.749 - 0.827) * t;
      b = 0.933 + (0.141 - 0.933) * t;
    } else {
      const t = (s - 0.5) * 2.0;
      // #fbbf24 (0.984, 0.749, 0.141) -> #f43f5e (0.957, 0.247, 0.369)
      r = 0.984 + (0.957 - 0.984) * t;
      g = 0.749 + (0.247 - 0.749) * t;
      b = 0.141 + (0.369 - 0.141) * t;
    }
    outArr[offset] = r;
    outArr[offset + 1] = g;
    outArr[offset + 2] = b;
  }

  // ===========================================================================
  // 4. GLSL ES 3.00 Shader Suite (Lit City, Instanced 3D Vehicles, Decals, Arcs)
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
  uniform vec3 u_focusTarget;
  uniform vec3 u_sceneCenter;
  uniform float u_fogScale;
  uniform float u_occludeRadius;
  uniform int u_passMode;

  out vec4 fragColor;

  void main() {
    float matKind = v_meta.x;

    // Emissive traffic signal stop-bars & overhead signal lenses
    if (matKind > 2.5) {
      if (u_passMode == 1) discard;
      fragColor = vec4(v_color.rgb, 1.0);
      return;
    }

    // Shader-computed anti-aliased UV-dashed white lane dividers (zero CPU while-loop)
    if (matKind > 0.45 && matKind < 0.55) {
      if (u_passMode == 1) discard;
      float sCoord = v_meta.y;
      float phase = fract(sCoord / 6.2) * 6.2;
      float aa = clamp(fwidth(sCoord), 0.02, 0.35);
      float dashMask = (1.0 - smoothstep(2.65 - aa, 2.65 + aa, phase)) * smoothstep(0.0, aa, phase);
      if (dashMask < 0.15) discard;
    }

    // Ground obsidian-navy baseplate (#080c14) with procedural distance-faded architectural grid
    if (matKind < 0.05) {
      if (u_passMode == 1) discard;
      vec3 groundCol = vec3(0.031, 0.047, 0.078);
      vec2 gp10 = v_worldPos.xy / 12.5;
      vec2 gp50 = v_worldPos.xy / 50.0;
      vec2 fw10 = max(fwidth(gp10), vec2(1e-4));
      vec2 fw50 = max(fwidth(gp50), vec2(1e-4));
      vec2 d10 = abs(fract(gp10 - 0.5) - 0.5) / fw10;
      vec2 d50 = abs(fract(gp50 - 0.5) - 0.5) / fw50;
      float line10 = 1.0 - min(min(d10.x, d10.y), 1.0);
      float line50 = 1.0 - min(min(d50.x, d50.y), 1.0);

      float camDist = length(v_worldPos - u_eyePos);
      float focusDist = length(v_worldPos.xy - u_focusTarget.xy);
      float gridFade = exp(-focusDist / max(320.0, u_fogScale * 0.72)) * clamp(1.25 - camDist / 1600.0, 0.0, 1.0);

      groundCol += vec3(0.020, 0.032, 0.052) * line10 * gridFade * 0.45;
      groundCol += vec3(0.038, 0.062, 0.098) * line50 * gridFade * 0.85;
      fragColor = vec4(groundCol, 1.0);
      return;
    }

    vec3 N = normalize(v_norm);
    vec3 L = normalize(vec3(0.44, -0.58, 0.68));
    vec3 Lfill = normalize(vec3(-0.55, 0.48, 0.68));
    vec3 V = normalize(u_eyePos - v_worldPos);
    vec3 H = normalize(L + V);

    vec3 base = v_color.rgb;
    float outAlpha = 1.0;

    // 3D Extruded Buildings (matKind == 1.0, exact height_m stored in v_color.a)
    if (matKind > 0.8 && matKind < 1.5) {
      float bldgH = v_color.a;
      float z = v_worldPos.z;

      // Near-focus camera-ray X-ray occlusion check ONLY when zoomed in close / following
      bool isOccluding = false;
      float dXY = 999.0;
      if (u_occludeRadius > 0.5 && bldgH > 9.5 && z > 2.0) {
        vec3 targetPt = u_focusTarget + vec3(0.0, 0.0, 1.6);
        vec3 seg = targetPt - u_eyePos;
        float segLenSq = max(1.0, dot(seg, seg));
        float tRay = dot(v_worldPos - u_eyePos, seg) / segLenSq;
        if (tRay > 0.14 && tRay < 0.88) {
          vec2 rayXY = u_eyePos.xy + seg.xy * tRay;
          dXY = length(v_worldPos.xy - rayXY);
          if (dXY < u_occludeRadius) {
            isOccluding = true;
          }
        }
      }

      if (u_passMode == 0 && isOccluding) {
        discard;
      }
      if (u_passMode == 1 && !isOccluding) {
        discard;
      }

      // Cool architectural twilight facade shading + subtle floor grooves + ground AO + roofline highlight
      if (abs(N.z) < 0.4) {
        float u = v_meta.y;
        if (z > 2.2 && (bldgH - z) > 0.55) {
          float floorCoord = z / 3.5;
          float colCoord = u / 3.4;
          float floorFrac = fract(floorCoord);
          float colFrac = fract(colCoord);
          float windowPane = smoothstep(0.22, 0.28, floorFrac) * (1.0 - smoothstep(0.74, 0.80, floorFrac)) *
                             smoothstep(0.16, 0.22, colFrac) * (1.0 - smoothstep(0.78, 0.84, colFrac));
          float fresnel = pow(1.0 - clamp(dot(N, V), 0.0, 1.0), 2.4);
          vec3 skyGlass = mix(vec3(0.08, 0.13, 0.21), vec3(0.24, 0.38, 0.56), fresnel);
          float cellHash = fract(sin(floor(floorCoord) * 19.17 + floor(colCoord) * 73.41) * 43758.5453);
          float warmInterior = step(0.88, cellHash);
          skyGlass = mix(skyGlass, vec3(0.85, 0.68, 0.42), warmInterior * 0.22);
          base = mix(base, skyGlass, windowPane * 0.65);
        }
      }

      // Ground-contact ambient occlusion gradient near z = 0
      float ao = mix(0.42, 1.0, smoothstep(0.0, 11.5, z));
      base *= ao;

      float ndotl = max(dot(N, L), 0.0);
      float fill = max(dot(N, Lfill), 0.0);
      float hemi = 0.5 * N.z + 0.5;
      vec3 ambient = mix(vec3(0.15, 0.19, 0.28), vec3(0.36, 0.46, 0.62), hemi);
      vec3 litBldg = base * (ambient + 0.72 * ndotl + 0.22 * fill);

      // Subtle bright architectural roofline edge highlight near z == bldgH
      float roofRim = smoothstep(0.48, 0.0, abs(bldgH - z));
      litBldg += vec3(0.16, 0.26, 0.40) * roofRim * (0.55 + 0.45 * hemi);

      float distB = length(v_worldPos.xy - u_sceneCenter.xy);
      float fogB = clamp((distB - u_fogScale * 0.72) / max(150.0, u_fogScale * 0.88), 0.0, 0.58);
      litBldg = mix(litBldg, vec3(0.031, 0.047, 0.078), fogB);

      if (u_passMode == 1) {
        float edgeAlpha = mix(0.42, 0.72, smoothstep(0.15 * u_occludeRadius, u_occludeRadius, dXY));
        fragColor = vec4(litBldg * 1.08, edgeAlpha);
        return;
      }

      fragColor = vec4(litBldg, outAlpha);
      return;
    }

    if (u_passMode == 1) discard;

    // Roads, curbs, parks, markings, signal gantries
    float ndotl = max(dot(N, L), 0.0);
    float hemi = 0.5 * N.z + 0.5;
    vec3 ambient = mix(vec3(0.24, 0.28, 0.36), vec3(0.52, 0.60, 0.72), hemi);
    vec3 lit = base * (ambient + 0.66 * ndotl);

    // Subtle twilight sheen on asphalt lanes & junctions
    if (matKind > 0.35 && matKind < 0.45 && N.z > 0.9) {
      float spec = pow(max(dot(N, H), 0.0), 28.0);
      lit += vec3(0.035, 0.050, 0.075) * spec;
    }

    float dist = length(v_worldPos.xy - u_sceneCenter.xy);
    float fog = clamp((dist - u_fogScale * 0.78) / max(160.0, u_fogScale * 0.90), 0.0, 0.55);
    lit = mix(lit, vec3(0.031, 0.047, 0.078), fog);

    fragColor = vec4(lit, 1.0);
  }`;

  // GPU-Instanced Sculpted 3D Vehicle Shader
  const VEH_VS = `#version 300 es
  precision highp float;
  layout(location = 0) in vec3 a_localPos;
  layout(location = 1) in vec3 a_localNorm;
  layout(location = 2) in float a_partId;

  // Per-instance attributes (divisor = 1)
  layout(location = 3) in vec4 a_iPose;  // x, y, yaw, speed
  layout(location = 4) in vec4 a_iDims;  // L, W, H, flags (bit0=isVan, bit1=isSelected)
  layout(location = 5) in vec4 a_iColor; // r, g, b, brakeIntensity

  uniform mat4 u_mvp;

  out vec3 v_worldPos;
  out vec3 v_norm;
  out vec3 v_bodyCol;
  out float v_partId;
  out float v_brake;
  out float v_selected;

  void main() {
    int flags = int(a_iDims.w + 0.5);
    bool isVan = (flags & 1) != 0;
    v_selected = (flags & 2) != 0 ? 1.0 : 0.0;

    vec3 lp = a_localPos;
    // Morph sedan cabin into full-length high-roof van greenhouse when isVan is set
    if (isVan && a_partId > 0.5 && a_partId < 1.5 && lp.z > 0.54) {
      lp.x = lp.x < 0.0 ? max(-0.47, lp.x * 1.38) : min(0.32, lp.x * 1.25);
      lp.y *= 1.15;
    }

    vec3 scaled = vec3(lp.x * a_iDims.x, lp.y * a_iDims.y, lp.z * a_iDims.z);
    float c = cos(a_iPose.z);
    float s = sin(a_iPose.z);

    vec3 worldPos = vec3(
      a_iPose.x + scaled.x * c - scaled.y * s,
      a_iPose.y + scaled.x * s + scaled.y * c,
      0.06 + scaled.z
    );

    vec3 invNorm = normalize(vec3(
      a_localNorm.x / max(0.5, a_iDims.x),
      a_localNorm.y / max(0.5, a_iDims.y),
      a_localNorm.z / max(0.5, a_iDims.z)
    ));
    vec3 worldNorm = vec3(
      invNorm.x * c - invNorm.y * s,
      invNorm.x * s + invNorm.y * c,
      invNorm.z
    );

    v_worldPos = worldPos;
    v_norm = worldNorm;
    v_bodyCol = a_iColor.rgb;
    v_partId = a_partId;
    v_brake = a_iColor.w;

    gl_Position = u_mvp * vec4(worldPos, 1.0);
  }`;

  const VEH_FS = `#version 300 es
  precision highp float;
  in vec3 v_worldPos;
  in vec3 v_norm;
  in vec3 v_bodyCol;
  in float v_partId;
  in float v_brake;
  in float v_selected;

  uniform vec3 u_eyePos;
  out vec4 fragColor;

  void main() {
    // Part 2: Warm-white LED headlight blades
    if (v_partId > 1.5 && v_partId < 2.5) {
      fragColor = vec4(0.98, 0.99, 1.0, 1.0);
      return;
    }
    // Part 3: Ruby LED tail & brake light strip
    if (v_partId > 2.5 && v_partId < 3.5) {
      vec3 ruby = mix(vec3(0.46, 0.06, 0.12), vec3(1.0, 0.13, 0.22), v_brake);
      fragColor = vec4(ruby, 1.0);
      return;
    }

    vec3 N = normalize(v_norm);
    vec3 L = normalize(vec3(0.44, -0.58, 0.68));
    vec3 V = normalize(u_eyePos - v_worldPos);
    vec3 H = normalize(L + V);
    float ndotl = max(dot(N, L), 0.0);
    float hemi = 0.5 * N.z + 0.5;

    // Part 4: Satin dark rubber tires
    if (v_partId > 3.5) {
      vec3 tire = vec3(0.06, 0.07, 0.09) * (0.45 + 0.55 * hemi + 0.35 * ndotl);
      fragColor = vec4(tire, 1.0);
      return;
    }

    // Part 1: Dark tinted glass cabin windows + body-tinted panoramic roof
    if (v_partId > 0.5 && v_partId < 1.5) {
      float specG = pow(max(dot(N, H), 0.0), 48.0);
      float fresnel = pow(1.0 - clamp(dot(N, V), 0.0, 1.0), 3.0);
      if (N.z > 0.82) {
        vec3 roofBase = mix(vec3(0.09, 0.14, 0.22), v_bodyCol, 0.62);
        vec3 roofLit = roofBase * (0.46 + 0.38 * hemi + 0.56 * ndotl) + vec3(0.28, 0.34, 0.42) * specG;
        fragColor = vec4(roofLit, 1.0);
        return;
      }
      vec3 glass = mix(vec3(0.06, 0.09, 0.15), vec3(0.22, 0.34, 0.48), fresnel * 0.75);
      glass += vec3(0.36, 0.42, 0.50) * specG;
      fragColor = vec4(glass, 1.0);
      return;
    }

    // Part 0: Sculpted automotive clearcoat body
    vec3 ambient = mix(vec3(0.26, 0.30, 0.38), vec3(0.56, 0.62, 0.72), hemi);
    vec3 lit = v_bodyCol * (ambient + 0.74 * ndotl);
    float spec = pow(max(dot(N, H), 0.0), 36.0);
    float rim = pow(1.0 - clamp(dot(N, V), 0.0, 1.0), 2.6);
    lit += vec3(0.30, 0.34, 0.40) * spec + v_bodyCol * rim * 0.18;
    if (v_selected > 0.5) {
      lit = mix(lit, vec3(1.0, 0.96, 0.84), 0.22);
    }
    fragColor = vec4(lit, 1.0);
  }`;

  // GPU-Instanced Ground Contact Shadow + Headlight Cone + Brake Light Pool
  const DECAL_VS = `#version 300 es
  precision highp float;
  layout(location = 0) in vec3 a_quad;   // u in [-1,1], v in [-1,1], kind (0=shadow, 1=headlight, 2=brake)
  layout(location = 3) in vec4 a_iPose;  // x, y, yaw, speed
  layout(location = 4) in vec4 a_iDims;  // L, W, H, flags
  layout(location = 5) in vec4 a_iColor; // r, g, b, brakeIntensity

  uniform mat4 u_mvp;
  out vec2 v_uv;
  out float v_kind;
  out float v_brake;

  void main() {
    v_uv = a_quad.xy;
    v_kind = a_quad.z;
    v_brake = a_iColor.w;

    float L = a_iDims.x;
    float W = a_iDims.y;
    float lx = 0.0;
    float ly = 0.0;
    float z = 0.062;

    if (v_kind < 0.5) {
      // Soft contact shadow directly under vehicle chassis
      lx = a_quad.x * (0.58 * L);
      ly = a_quad.y * (0.64 * W);
      z = 0.062;
    } else if (v_kind < 1.5) {
      // Warm-white headlight cone ahead of front bumper (u mapped 0..1 forward)
      float fwd = 0.5 * (a_quad.x + 1.0);
      lx = 0.48 * L + fwd * 5.2;
      ly = a_quad.y * (0.52 * W + fwd * 1.15);
      z = 0.068;
    } else {
      // Ruby brake glow behind rear bumper when braking/stopped
      float back = 0.5 * (a_quad.x + 1.0);
      lx = -0.48 * L - back * 2.5 * step(0.55, v_brake);
      ly = a_quad.y * (0.55 * W + back * 0.65);
      z = 0.068;
    }

    float c = cos(a_iPose.z);
    float s = sin(a_iPose.z);
    vec3 worldPos = vec3(
      a_iPose.x + lx * c - ly * s,
      a_iPose.y + lx * s + ly * c,
      z
    );
    gl_Position = u_mvp * vec4(worldPos, 1.0);
  }`;

  const DECAL_FS = `#version 300 es
  precision highp float;
  in vec2 v_uv;
  in float v_kind;
  in float v_brake;
  out vec4 fragColor;

  void main() {
    if (v_kind < 0.5) {
      vec2 d = abs(v_uv);
      float boxDist = length(max(d - vec2(0.56, 0.52), 0.0));
      float alpha = 0.54 * (1.0 - smoothstep(0.0, 0.44, boxDist));
      fragColor = vec4(0.01, 0.015, 0.03, alpha);
      return;
    }
    if (v_kind < 1.5) {
      float fwd = 0.5 * (v_uv.x + 1.0);
      float lat = exp(-2.6 * v_uv.y * v_uv.y);
      float longFade = (1.0 - smoothstep(0.10, 1.0, fwd)) * smoothstep(0.0, 0.12, fwd);
      float alpha = 0.13 * lat * longFade;
      fragColor = vec4(0.86, 0.94, 1.0, alpha);
      return;
    }
    if (v_brake < 0.55) discard;
    float back = 0.5 * (v_uv.x + 1.0);
    float latB = exp(-2.4 * v_uv.y * v_uv.y);
    float backFade = 1.0 - smoothstep(0.0, 1.0, back);
    fragColor = vec4(1.0, 0.14, 0.24, 0.24 * latB * backFade * v_brake);
  }`;

  // Luminous 3D Parabolic Causal DBN Arcs, Signal Asphalt Light Pools & Corridor Glow
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

    // Mode -2.0: Radial soft asphalt light pool (traffic signal colored pool)
    if (mode < -1.5) {
      float r = length(v_uv);
      if (r > 1.0) discard;
      float falloff = pow(1.0 - r, 2.1);
      fragColor = vec4(v_color.rgb, v_color.a * falloff);
      return;
    }

    // Mode -1.0: Crisp glowing ring / corridor spillback border ribbon
    if (mode < -0.5) {
      float edgeSoft = 1.0 - smoothstep(0.35, 1.0, abs(v_uv.y));
      fragColor = vec4(v_color.rgb, v_color.a * edgeSoft);
      return;
    }

    // Sleek 3D Parabolic Causal DBN Arc with directional energy pulse src(u=0) -> dst(u=1)
    float core = exp(-2.0 * v_uv.y * v_uv.y);
    float endFade = smoothstep(0.0, 0.07, v_uv.x) * (1.0 - smoothstep(0.93, 1.0, v_uv.x));
    float speed = 0.78 + 1.45 * clamp(v_pulse.y, 0.0, 1.0);
    float phase = fract(v_uv.x * 1.40 - u_animTime * speed);
    // Asymmetric comet pulse: bright leading front, smooth trailing wake
    float pulse = exp(-11.0 * pow(phase - 0.55, 2.0)) + 0.42 * exp(-4.5 * pow(phase - 0.36, 2.0));
    float pClamped = clamp(pulse, 0.0, 1.0);

    vec3 col = mix(v_color.rgb, vec3(1.0, 1.0, 1.0), 0.62 * pClamped);
    float alpha = v_color.a * core * (0.58 + 0.42 * pClamped) * (0.36 + 0.64 * endFade);
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
  // 5. Geometry Builders & Sculpted 3D Vehicle Template Mesh
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
      const s0 = u0 !== undefined ? u0 : 0.0;
      const s1 = u1 !== undefined ? u1 : 0.0;
      this.pushVertex(p0[0], p0[1], p0[2], n[0], n[1], n[2], r, g, b, a, m0, s0);
      this.pushVertex(p1[0], p1[1], p1[2], n[0], n[1], n[2], r, g, b, a, m0, s1);
      this.pushVertex(p2[0], p2[1], p2[2], n[0], n[1], n[2], r, g, b, a, m0, s1);

      this.pushVertex(p0[0], p0[1], p0[2], n[0], n[1], n[2], r, g, b, a, m0, s0);
      this.pushVertex(p2[0], p2[1], p2[2], n[0], n[1], n[2], r, g, b, a, m0, s1);
      this.pushVertex(p3[0], p3[1], p3[2], n[0], n[1], n[2], r, g, b, a, m0, s0);
    }

    pushRibbon2D(pts, width, z, col, m0, trackArcLength) {
      if (!pts || pts.length < 2) return;
      const hw = 0.5 * width;
      let sAcc = 0.0;
      for (let i = 0; i < pts.length - 1; i++) {
        const x0 = pts[i][0], y0 = pts[i][1];
        const x1 = pts[i + 1][0], y1 = pts[i + 1][1];
        const dx = x1 - x0, dy = y1 - y0;
        const segLen = Math.hypot(dx, dy);
        if (segLen < 1e-4) continue;
        const nx = (-dy / segLen) * hw;
        const ny = (dx / segLen) * hw;
        const u0 = trackArcLength ? sAcc : 0.0;
        const u1 = trackArcLength ? sAcc + segLen : 0.0;
        this.pushQuad(
          [x0 - nx, y0 - ny, z],
          [x1 - nx, y1 - ny, z],
          [x1 + nx, y1 + ny, z],
          [x0 + nx, y0 + ny, z],
          [0, 0, 1],
          col,
          m0 || 0.0,
          u0,
          u1
        );
        sAcc += segLen;
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

      this.pushQuad(t0, t1, t2, t3, [0, 0, 1], col, m0, 0, 0);
      this.pushQuad(b1, b2, t2, t1, [c, s, 0.15], col, m0, 0, 0);
      this.pushQuad(b3, b0, t0, t3, [-c, -s, 0.15], col, m0, 0, 0);
      this.pushQuad(b0, b1, t1, t0, [s, -c, 0.15], col, m0, 0, 0);
      this.pushQuad(b2, b3, t3, t2, [-s, c, 0.15], col, m0, 0, 0);
    }

    get vertexCount() {
      return (this.len / 12) | 0;
    }
  }

  class ArcMeshBuilder {
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

    pushLightPoolDisc(cx, cy, z, radius, col) {
      const r = col[0], g = col[1], b = col[2], a = col[3];
      const x0 = cx - radius, x1 = cx + radius;
      const y0 = cy - radius, y1 = cy + radius;
      this.pushVertex(x0, y0, z, -1.0, -1.0, r, g, b, a, 1.0, -2.0);
      this.pushVertex(x1, y0, z, 1.0, -1.0, r, g, b, a, 1.0, -2.0);
      this.pushVertex(x1, y1, z, 1.0, 1.0, r, g, b, a, 1.0, -2.0);

      this.pushVertex(x0, y0, z, -1.0, -1.0, r, g, b, a, 1.0, -2.0);
      this.pushVertex(x1, y1, z, 1.0, 1.0, r, g, b, a, 1.0, -2.0);
      this.pushVertex(x0, y1, z, -1.0, 1.0, r, g, b, a, 1.0, -2.0);
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
          0.0,
          1.0,
          col,
          1.0,
          -1.0
        );
      }
    }

    pushParabolicArc3D(x0, y0, x1, y1, zBase, apexH, ribbonHalfW, col, omega, gamma, eyePos) {
      const dx = x1 - x0, dy = y1 - y0;
      const dist = Math.hypot(dx, dy);
      if (dist < 1.8) return;
      const steps = dist > 85.0 ? 18 : 14;

      let prevX = x0, prevY = y0, prevZ = zBase;
      for (let k = 1; k <= steps; k++) {
        const u0 = (k - 1) / steps;
        const u1 = k / steps;
        const currX = x0 + dx * u1;
        const currY = y0 + dy * u1;
        const currZ = zBase + 4.0 * apexH * u1 * (1.0 - u1);

        const taper0 = 0.62 + 0.38 * Math.sin(u0 * Math.PI);
        const taper1 = 0.62 + 0.38 * Math.sin(u1 * Math.PI);

        const tx = currX - prevX, ty = currY - prevY, tz = currZ - prevZ;
        const vx = eyePos[0] - currX, vy = eyePos[1] - currY, vz = eyePos[2] - currZ;
        let bx = ty * vz - tz * vy;
        let by = tz * vx - tx * vz;
        let bz = tx * vy - ty * vx;
        const blen = Math.hypot(bx, by, bz) || 1.0;
        bx /= blen; by /= blen; bz /= blen;

        const w0 = ribbonHalfW * taper0;
        const w1 = ribbonHalfW * taper1;

        this.pushQuadUV(
          [prevX - bx * w0, prevY - by * w0, prevZ - bz * w0],
          [currX - bx * w1, currY - by * w1, currZ - bz * w1],
          [currX + bx * w1, currY + by * w1, currZ + bz * w1],
          [prevX + bx * w0, prevY + by * w0, prevZ + bz * w0],
          u0,
          u1,
          col,
          omega,
          gamma
        );
        prevX = currX; prevY = currY; prevZ = currZ;
      }
    }

    get vertexCount() {
      return (this.len / 11) | 0;
    }
  }

  function buildSculptedVehicleTemplate() {
    // 7 floats per vertex: [lx, ly, lz, nx, ny, nz, partId]
    const verts = [];
    const pushTri = (a, b, c, nx, ny, nz, partId) => {
      verts.push(a[0], a[1], a[2], nx, ny, nz, partId);
      verts.push(b[0], b[1], b[2], nx, ny, nz, partId);
      verts.push(c[0], c[1], c[2], nx, ny, nz, partId);
    };
    const pushQuadAuto = (p0, p1, p2, p3, partId) => {
      const ux = p1[0] - p0[0], uy = p1[1] - p0[1], uz = p1[2] - p0[2];
      const vx = p2[0] - p0[0], vy = p2[1] - p0[1], vz = p2[2] - p0[2];
      let nx = uy * vz - uz * vy;
      let ny = uz * vx - ux * vz;
      let nz = ux * vy - uy * vx;
      const len = Math.hypot(nx, ny, nz) || 1.0;
      nx /= len; ny /= len; nz /= len;
      pushTri(p0, p1, p2, nx, ny, nz, partId);
      pushTri(p0, p2, p3, nx, ny, nz, partId);
    };
    const pushLoftSection = (b0, b1, b2, b3, t0, t1, t2, t3, partId, includeTop) => {
      // b0=(-x,-y), b1=(+x,-y), b2=(+x,+y), b3=(-x,+y)
      if (includeTop) pushQuadAuto(t0, t1, t2, t3, partId);
      pushQuadAuto(b1, b2, t2, t1, partId); // Front (+X)
      pushQuadAuto(b3, b0, t0, t3, partId); // Rear (-X)
      pushQuadAuto(b0, b1, t1, t0, partId); // Right (-Y)
      pushQuadAuto(b2, b3, t3, t2, partId); // Left (+Y)
    };
    const pushBox = (x0, x1, y0, y1, z0, z1, partId, includeTop) => {
      pushLoftSection(
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
        partId,
        includeTop !== false
      );
    };

    // 1. Four Tires / Wheels (partId = 4)
    for (const ax of [-0.30, 0.30]) {
      for (const ay of [-0.46, 0.46]) {
        pushBox(ax - 0.085, ax + 0.085, ay - 0.055, ay + 0.055, 0.0, 0.24, 4.0, false);
      }
    }

    // 2. Lower Sculpted Chassis Skirt (partId = 0, z = 0.09..0.28)
    pushLoftSection(
      [-0.47, -0.45, 0.09], [0.47, -0.44, 0.09], [0.47, 0.44, 0.09], [-0.47, 0.45, 0.09],
      [-0.50, -0.50, 0.28], [0.50, -0.48, 0.28], [0.50, 0.48, 0.28], [-0.50, 0.50, 0.28],
      0.0,
      false
    );

    // 3. Upper Aerodynamic Body / Hood / Rear Deck (partId = 0, z = 0.28..0.56)
    pushLoftSection(
      [-0.50, -0.50, 0.28], [0.50, -0.48, 0.28], [0.50, 0.48, 0.28], [-0.50, 0.50, 0.28],
      [-0.48, -0.46, 0.56], [0.46, -0.44, 0.53], [0.46, 0.44, 0.53], [-0.48, 0.46, 0.56],
      0.0,
      true
    );

    // 4. Raked Glass Greenhouse Cabin (partId = 1, z = 0.54..1.00)
    pushLoftSection(
      [-0.36, -0.43, 0.54], [0.24, -0.43, 0.54], [0.24, 0.43, 0.54], [-0.36, 0.43, 0.54],
      [-0.25, -0.35, 1.00], [0.05, -0.35, 1.00], [0.05, 0.35, 1.00], [-0.25, 0.35, 1.00],
      1.0,
      true
    );

    // 5. Front LED Headlight Blades (partId = 2)
    pushBox(0.472, 0.508, -0.44, -0.20, 0.34, 0.46, 2.0, true);
    pushBox(0.472, 0.508, 0.20, 0.44, 0.34, 0.46, 2.0, true);

    // 6. Full-Width Rear Ruby LED Tail/Brake Light Bar (partId = 3)
    pushBox(-0.508, -0.474, -0.44, 0.44, 0.39, 0.49, 3.0, true);

    const floatData = new Float32Array(verts);
    return {
      data: floatData,
      vertexCount: (verts.length / 7) | 0,
    };
  }

  function buildDecalQuadBuffer() {
    // 3 quads (kind = 0 shadow, 1 headlight cone, 2 brake pool) x 6 vertices x 3 floats
    const out = [];
    for (let kind = 0; kind < 3; kind++) {
      out.push(-1, -1, kind, 1, -1, kind, 1, 1, kind);
      out.push(-1, -1, kind, 1, 1, kind, -1, 1, kind);
    }
    return new Float32Array(out);
  }

  // ===========================================================================
  // 6. Static Urban Scene Compilation (Twilight Architecture + Signal Stop-Bars)
  // ===========================================================================

  function buildStaticCityBuffers(scene) {
    const roadBuilder = new LitMeshBuilder(262144);
    const buildingBuilder = new LitMeshBuilder(524288);

    const b = scene.bounds;
    const pad = Math.max(420.0, 0.65 * Math.max(b.span_x, b.span_y));
    const gx0 = b.min_x - pad, gx1 = b.max_x + pad;
    const gy0 = b.min_y - pad, gy1 = b.max_y + pad;

    // 1. Deep obsidian-navy ground plane (#080c14) with shader-evaluated anti-aliased grid
    roadBuilder.pushQuad(
      [gx0, gy0, -0.05],
      [gx1, gy0, -0.05],
      [gx1, gy1, -0.05],
      [gx0, gy1, -0.05],
      [0, 0, 1],
      [0.031, 0.047, 0.078, 1.0],
      0.0
    );

    // 2. Muted sage-slate urban parks & plazas
    const parks = scene.parks || [];
    for (let p = 0; p < parks.length; p++) {
      const pk = parks[p];
      const fp = pk.footprint;
      const tris = pk.triangles || [];
      const col = [0.075, 0.145, 0.125];
      for (let t = 0; t + 2 < tris.length; t += 3) {
        const a = fp[tris[t]], bPt = fp[tris[t + 1]], c = fp[tris[t + 2]];
        if (!a || !bPt || !c) continue;
        roadBuilder.pushVertex(a[0], a[1], 0.008, 0, 0, 1, col[0], col[1], col[2], 1.0, 0.2, 0.0);
        roadBuilder.pushVertex(bPt[0], bPt[1], 0.008, 0, 0, 1, col[0], col[1], col[2], 1.0, 0.2, 0.0);
        roadBuilder.pushVertex(c[0], c[1], 0.008, 0, 0, 1, col[0], col[1], col[2], 1.0, 0.2, 0.0);
      }
    }

    // 3. Multi-lane rich dark asphalt + crisp sidewalk curbs
    const lanes = scene.lanes || [];
    const laneById = new Map();
    const curbCol = [0.17, 0.22, 0.30, 1.0];
    const asphaltCol = [0.105, 0.132, 0.185, 1.0];
    for (let i = 0; i < lanes.length; i++) {
      const ln = lanes[i];
      laneById.set(ln.lane_id, ln);
      roadBuilder.pushRibbon2D(ln.shape, ln.width_m + 0.62, 0.025, curbCol, 0.4, false);
      roadBuilder.pushRibbon2D(ln.shape, ln.width_m * 0.985, 0.045, asphaltCol, 0.4, false);
    }

    // 4. Triangulated junction asphalt boxes
    const junctions = scene.junctions || [];
    const juncCol = [0.118, 0.148, 0.205, 1.0];
    for (let j = 0; j < junctions.length; j++) {
      const jn = junctions[j];
      const poly = jn.polygon || [];
      const tris = jn.triangles || [];
      if (poly.length >= 3 && tris.length >= 3) {
        for (let t = 0; t + 2 < tris.length; t += 3) {
          const p0 = poly[tris[t]], p1 = poly[tris[t + 1]], p2 = poly[tris[t + 2]];
          if (!p0 || !p1 || !p2) continue;
          roadBuilder.pushVertex(p0[0], p0[1], 0.050, 0, 0, 1, juncCol[0], juncCol[1], juncCol[2], 1.0, 0.4, 0.0);
          roadBuilder.pushVertex(p1[0], p1[1], 0.050, 0, 0, 1, juncCol[0], juncCol[1], juncCol[2], 1.0, 0.4, 0.0);
          roadBuilder.pushVertex(p2[0], p2[1], 0.050, 0, 0, 1, juncCol[0], juncCol[1], juncCol[2], 1.0, 0.4, 0.0);
        }
      }
    }

    // 5. Crisp lane dividers (shader UV-dashed white + warm amber center lines), stop bars, crosswalks, arrows
    const dividers = scene.lane_dividers || [];
    for (let i = 0; i < dividers.length; i++) {
      const dv = dividers[i];
      if (dv.kind === "yellow_center") {
        roadBuilder.pushRibbon2D(dv.points, 0.17, 0.068, [0.90, 0.68, 0.20, 1.0], 0.4, false);
      } else {
        // matKind = 0.5 evaluates anti-aliased dashes in LIT_FS using arc-length in a_meta.y
        roadBuilder.pushRibbon2D(dv.points, 0.12, 0.068, [0.74, 0.82, 0.92, 1.0], 0.5, true);
      }
    }

    const stopBars = scene.stop_bars || [];
    for (let i = 0; i < stopBars.length; i++) {
      const sb = stopBars[i];
      roadBuilder.pushRibbon2D([sb.p0, sb.p1], 0.36, 0.072, [0.58, 0.66, 0.78, 1.0], 0.4, false);
    }

    const crosswalks = scene.crosswalks || [];
    for (let i = 0; i < crosswalks.length; i++) {
      const cw = crosswalks[i];
      const yaw = cw.yaw_ccw_rad;
      const cx = cw.xy[0], cy = cw.xy[1];
      const nx = -Math.sin(yaw), ny = Math.cos(yaw);
      const fx = Math.cos(yaw), fy = Math.sin(yaw);
      const nStripes = Math.max(3, Math.floor(cw.width_m / 1.25));
      const span = cw.width_m * 0.84;
      for (let s = 0; s < nStripes; s++) {
        const lat = -0.5 * span + (nStripes > 1 ? (s / (nStripes - 1)) * span : 0.0);
        const sx = cx + nx * lat, sy = cy + ny * lat;
        roadBuilder.pushRibbon2D(
          [[sx - fx * 0.92, sy - fy * 0.92], [sx + fx * 0.92, sy + fy * 0.92]],
          0.38,
          0.070,
          [0.52, 0.60, 0.72, 1.0],
          0.4,
          false
        );
      }
    }

    const arrows = scene.turn_arrows || [];
    const arrowCol = [0.46, 0.54, 0.66, 1.0];
    for (let i = 0; i < arrows.length; i++) {
      const ar = arrows[i];
      const cx = ar.xy[0], cy = ar.xy[1], yaw = ar.yaw_ccw_rad;
      const fx = Math.cos(yaw), fy = Math.sin(yaw);
      const lx = -fy, ly = fx;
      roadBuilder.pushRibbon2D(
        [[cx - fx * 1.2, cy - fy * 1.2], [cx + fx * 0.75, cy + fy * 0.75]],
        0.18,
        0.071,
        arrowCol,
        0.4,
        false
      );
      if (ar.dirs.indexOf("S") >= 0) {
        roadBuilder.pushRibbon2D(
          [[cx + fx * 0.45, cy + fy * 0.45], [cx + fx * 1.45, cy + fy * 1.45]],
          0.28,
          0.071,
          arrowCol,
          0.4,
          false
        );
      }
      if (ar.dirs.indexOf("L") >= 0) {
        roadBuilder.pushRibbon2D(
          [[cx + fx * 0.35, cy + fy * 0.35], [cx + fx * 0.85 + lx * 0.78, cy + fy * 0.85 + ly * 0.78]],
          0.22,
          0.071,
          arrowCol,
          0.4,
          false
        );
      }
      if (ar.dirs.indexOf("R") >= 0) {
        roadBuilder.pushRibbon2D(
          [[cx + fx * 0.35, cy + fy * 0.35], [cx + fx * 0.85 - lx * 0.78, cy + fy * 0.85 - ly * 0.78]],
          0.22,
          0.071,
          arrowCol,
          0.4,
          false
        );
      }
    }

    // 6. Static 3D Traffic Signal Poles, Cantilever Mast Arms & Pre-Indexed Glowing Stop-Bar Descriptors
    const signals = scene.signals || [];
    const poleCol = [0.22, 0.27, 0.36, 1.0];
    const housingCol = [0.10, 0.13, 0.18, 1.0];
    const signalLaneDescriptors = [];

    for (let s = 0; s < signals.length; s++) {
      const sig = signals[s];
      const poles = sig.poles || [];
      const poleByEdge = new Map();
      for (let p = 0; p < poles.length; p++) {
        const pl = poles[p];
        poleByEdge.set(pl.edge_id, pl);
        roadBuilder.pushOrientedBox(
          pl.base_xy[0],
          pl.base_xy[1],
          0.0,
          pl.height_m || 5.7,
          pl.yaw_ccw_rad || 0.0,
          0.15,
          0.15,
          poleCol,
          0.4,
          0.86
        );
      }

      const byInLane = new Map();
      const links = sig.links || [];
      for (let l = 0; l < links.length; l++) {
        const lk = links[l];
        const hx = lk.head_xyz[0], hy = lk.head_xyz[1], hz = lk.head_xyz[2];
        roadBuilder.pushOrientedBox(hx, hy, hz - 0.30, hz + 0.30, lk.yaw_ccw_rad, 0.18, 0.26, housingCol, 0.4, 1.0);

        let group = byInLane.get(lk.in_lane_id);
        if (!group) {
          const lnObj = laneById.get(lk.in_lane_id);
          const widthM = lnObj ? lnObj.width_m : 3.3;
          // Incoming travel heading is opposite of signal face yaw
          const travelYaw = lk.yaw_ccw_rad - Math.PI;
          group = {
            tlsId: sig.tls_id,
            edgeId: lk.edge_id,
            cx: hx,
            cy: hy,
            headZ: hz,
            faceYaw: lk.yaw_ccw_rad,
            travelYaw,
            widthM,
            linkIndices: [],
          };
          byInLane.set(lk.in_lane_id, group);
        }
        group.linkIndices.push(lk.link_index);
      }

      // Draw sleek overhead cantilever mast arm from curbside pole out to farthest controlled lane head
      const groupsArr = Array.from(byInLane.values());
      for (let g = 0; g < groupsArr.length; g++) {
        signalLaneDescriptors.push(groupsArr[g]);
      }

      for (const [edgeId, pl] of poleByEdge.entries()) {
        let farthest = null;
        let maxDist = 0.0;
        for (let g = 0; g < groupsArr.length; g++) {
          if (groupsArr[g].edgeId === edgeId) {
            const d = Math.hypot(groupsArr[g].cx - pl.base_xy[0], groupsArr[g].cy - pl.base_xy[1]);
            if (d > maxDist) {
              maxDist = d;
              farthest = groupsArr[g];
            }
          }
        }
        if (farthest && maxDist > 0.5) {
          const mx = 0.5 * (pl.base_xy[0] + farthest.cx);
          const my = 0.5 * (pl.base_xy[1] + farthest.cy);
          const armYaw = Math.atan2(farthest.cy - pl.base_xy[1], farthest.cx - pl.base_xy[0]);
          roadBuilder.pushOrientedBox(mx, my, 5.42, 5.56, armYaw, 0.5 * maxDist + 0.15, 0.08, poleCol, 0.4, 1.0);
        }
      }
    }

    // 7. Architectural Twilight 3D Extruded Buildings (height_m stored in alpha for shader AO/roofline/X-ray)
    const buildings = scene.buildings || [];
    for (let bIdx = 0; bIdx < buildings.length; bIdx++) {
      const bldg = buildings[bIdx];
      const fp = bldg.footprint;
      const tris = bldg.triangles || [];
      const h = Math.max(4.0, bldg.height_m || 12.0);
      const srcCol = bldg.color_rgb || [0.74, 0.78, 0.84];

      // Remap daytime pastel tint into cohesive cool twilight slate-indigo architecture
      const wr = 0.13 + 0.14 * srcCol[0];
      const wg = 0.17 + 0.15 * srcCol[1];
      const wb = 0.25 + 0.18 * srcCol[2];
      const wallCol = [wr, wg, wb, h];
      const roofCol = [wr * 0.86, wg * 0.89, wb * 0.94, h];

      let perimU = 0.0;
      for (let i = 0; i < fp.length; i++) {
        const p0 = fp[i];
        const p1 = fp[(i + 1) % fp.length];
        const dx = p1[0] - p0[0], dy = p1[1] - p0[1];
        const segLen = Math.hypot(dx, dy);
        if (segLen < 0.05) continue;
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

      for (let t = 0; t + 2 < tris.length; t += 3) {
        const a = fp[tris[t]], bPt = fp[tris[t + 1]], c = fp[tris[t + 2]];
        if (!a || !bPt || !c) continue;
        buildingBuilder.pushVertex(a[0], a[1], h, 0, 0, 1, roofCol[0], roofCol[1], roofCol[2], h, 1.0, 0.0);
        buildingBuilder.pushVertex(bPt[0], bPt[1], h, 0, 0, 1, roofCol[0], roofCol[1], roofCol[2], h, 1.0, 0.0);
        buildingBuilder.pushVertex(c[0], c[1], h, 0, 0, 1, roofCol[0], roofCol[1], roofCol[2], h, 1.0, 0.0);
      }
    }

    // Pre-compile corridor border rib segments for zero-alloc spillback glow
    const corridorBorders = new Map();
    const ribbons = scene.corridor_ribbons || {};
    const rEntries = Object.entries(ribbons);
    for (let i = 0; i < rEntries.length; i++) {
      const cid = rEntries[i][0];
      const shapes = rEntries[i][1];
      if (!shapes || shapes.length === 0) continue;
      // Pick outermost right/left lanes of the corridor for clean border glow
      const outerShapes = shapes.length > 1 ? [shapes[0], shapes[shapes.length - 1]] : [shapes[0]];
      corridorBorders.set(cid, outerShapes);
    }

    return {
      roadFloats: roadBuilder.data.subarray(0, roadBuilder.len),
      roadVertexCount: roadBuilder.vertexCount,
      buildingFloats: buildingBuilder.data.subarray(0, buildingBuilder.len),
      buildingVertexCount: buildingBuilder.vertexCount,
      signalLaneDescriptors,
      corridorBorders,
    };
  }

  // ===========================================================================
  // 7. Zero-Allocation Pre-Indexed Vehicle Slot Table (60-120 Hz Interpolation)
  // ===========================================================================

  function buildIndexedTrace(trace) {
    const idToSlot = new Map();
    const slotIds = [];
    const slotVtypes = [];
    const slotIsVan = [];

    // Fast zero-object-allocation path for @viz-codec's columnar Structure-of-Arrays (soa_v2) format
    if (!Array.isArray(trace.frames) && trace.encoding === "soa_v2" && Array.isArray(trace.packed_frames)) {
      const strings = trace.strings || [];
      const vehMeta = trace.veh_meta || [];
      const tlsKeys = trace.tls_keys || [];
      const pFrames = trace.packed_frames;

      for (let s = 0; s < vehMeta.length; s++) {
        const vm = vehMeta[s];
        const id = vm[0];
        const vtype = strings[vm[1]] || "passenger";
        const lenM = vm[2] || 4.5;
        idToSlot.set(id, s);
        slotIds.push(id);
        slotVtypes.push(vtype);
        const vt = String(vtype).toLowerCase();
        slotIsVan.push(lenM > 5.4 || vt.indexOf("van") >= 0 || vt.indexOf("truck") >= 0 || vt.indexOf("bus") >= 0 ? 1 : 0);
      }

      const emptyI32 = new Int32Array(0);
      const emptyF32 = new Float32Array(0);
      const emptyI8 = new Int8Array(0);
      const emptyU8 = new Uint8Array(0);

      const indexedFrames = new Array(pFrames.length);
      for (let fIdx = 0; fIdx < pFrames.length; fIdx++) {
        const pf = pFrames[fIdx];
        const t = pf[0];
        const vCols = pf[1];
        const eCols = pf[2];
        const hCols = pf[3];
        const tlsVals = pf[4];
        const kArr = pf[6];

        let n = 0;
        let slot = emptyI32, x = emptyF32, y = emptyF32, yaw = emptyF32, v = emptyF32;
        let L = emptyF32, W = emptyF32, dPost = emptyF32, psi = emptyF32;
        let mode = emptyI8, brake = emptyU8;

        if (vCols && vCols.length > 0) {
          const vIdx = vCols[0];
          const xCm = vCols[1];
          const yCm = vCols[2];
          const yawMr = vCols[3];
          const vCms = vCols[4];
          const sBits = vCols[7];
          const dPm = vCols[8];
          const psiPm = vCols[9];
          n = vIdx.length;
          const hasD = Boolean(dPm && dPm.length > 0);
          const hasPsi = Boolean(psiPm && psiPm.length > 0);

          slot = new Int32Array(n);
          x = new Float32Array(n);
          y = new Float32Array(n);
          yaw = new Float32Array(n);
          v = new Float32Array(n);
          L = new Float32Array(n);
          W = new Float32Array(n);
          dPost = new Float32Array(n);
          psi = new Float32Array(n);
          mode = new Int8Array(n);
          brake = new Uint8Array(n);

          for (let i = 0; i < n; i++) {
            const s = vIdx[i];
            const vm = vehMeta[s];
            const sb = sBits[i];
            slot[i] = s;
            x[i] = xCm[i] * 0.01;
            y[i] = yCm[i] * 0.01;
            yaw[i] = yawMr[i] * 0.001;
            v[i] = vCms[i] * 0.01;
            L[i] = vm ? (vm[2] || 4.5) : 4.5;
            W[i] = vm ? (vm[3] || 1.85) : 1.85;
            dPost[i] = hasD ? dPm[i] * 0.001 : 0.0;
            psi[i] = hasPsi ? psiPm[i] * 0.001 : 0.0;
            mode[i] = ((sb >> 4) & 15) - 1;
            brake[i] = sb & 1;
          }
        }

        let validEdges = 0;
        let eSrc = emptyI32, eDst = emptyI32, eRel = emptyU8, eOmega = emptyF32, eGamma = emptyF32;
        if (eCols && eCols.length > 0) {
          const eSrcCol = eCols[0];
          const eDstCol = eCols[1];
          const eRelCol = eCols[2];
          const eOmCol = eCols[3];
          const eGaCol = eCols[5];
          const m = eSrcCol.length;
          const hasG = Boolean(eGaCol && eGaCol.length > 0);

          eSrc = new Int32Array(m);
          eDst = new Int32Array(m);
          eRel = new Uint8Array(m);
          eOmega = new Float32Array(m);
          eGamma = new Float32Array(m);

          for (let e = 0; e < m; e++) {
            eSrc[e] = eSrcCol[e];
            eDst[e] = eDstCol[e];
            eRel[e] = Math.max(0, Math.min(3, eRelCol[e] | 0));
            const om = eOmCol[e] * 0.001;
            eOmega[e] = om;
            eGamma[e] = hasG ? eGaCol[e] * 0.001 : om;
          }
          validEdges = m;
        }

        const heatArr = [];
        if (hCols && hCols.length > 0) {
          const hCid = hCols[0];
          const hTail = hCols[2];
          const hSpill = hCols[3];
          for (let h = 0, hn = hCid.length; h < hn; h++) {
            const spill = Math.max(hSpill[h] * 0.001, hTail[h] * 0.001 * 0.92);
            if (spill > 0.35) {
              heatArr.push({ cid: strings[hCid[h]], spill });
            }
          }
        }

        const tls = {};
        if (tlsVals) {
          for (let k = 0; k < tlsKeys.length && k < tlsVals.length; k++) {
            tls[tlsKeys[k]] = tlsVals[k];
          }
        }

        const kpi = kArr && kArr.length >= 10 ? {
          active_veh: kArr[0],
          halted_veh: kArr[1],
          mean_speed_mps: kArr[2],
          active_edges: kArr[3],
          mean_psi: kArr[4],
          spillback_corridors: kArr[5],
          cum_collisions: kArr[6],
          cum_emerg_brakes: kArr[7],
          cum_reroutes: kArr[8],
          dbn_ms: kArr[9],
        } : null;

        indexedFrames[fIdx] = {
          raw: null,
          t,
          n,
          slot,
          x,
          y,
          yaw,
          v,
          L,
          W,
          dPost,
          psi,
          mode,
          brake,
          edgeCount: validEdges,
          eSrc,
          eDst,
          eRel,
          eOmega,
          eGamma,
          heatArr,
          tls,
          kpi,
        };
      }

      const tStart = indexedFrames.length > 0 ? indexedFrames[0].t : 0.0;
      const tEnd = indexedFrames.length > 0 ? indexedFrames[indexedFrames.length - 1].t : 0.0;
      const dt = trace.dt || (indexedFrames.length >= 2 ? Math.max(0.05, indexedFrames[1].t - indexedFrames[0].t) : 0.5);

      return {
        raw: trace,
        frames: indexedFrames,
        numSlots: Math.max(1, slotIds.length),
        idToSlot,
        slotIds,
        slotVtypes,
        slotIsVan: new Uint8Array(slotIsVan),
        tStart,
        tEnd,
        duration: Math.max(0.0, tEnd - tStart),
        dt,
      };
    }

    const frames = trace.frames || [];
    const ensureSlot = (id, vtype, L) => {
      let s = idToSlot.get(id);
      if (s === undefined) {
        s = slotIds.length;
        idToSlot.set(id, s);
        slotIds.push(id);
        slotVtypes.push(vtype || "passenger");
        const vt = String(vtype || "").toLowerCase();
        slotIsVan.push(L > 5.4 || vt.indexOf("van") >= 0 || vt.indexOf("truck") >= 0 || vt.indexOf("bus") >= 0 ? 1 : 0);
      }
      return s;
    };

    const indexedFrames = new Array(frames.length);
    for (let fIdx = 0; fIdx < frames.length; fIdx++) {
      const f = frames[fIdx];
      const vs = f.vehicles || [];
      const n = vs.length;

      const slot = new Int32Array(n);
      const x = new Float32Array(n);
      const y = new Float32Array(n);
      const yaw = new Float32Array(n);
      const v = new Float32Array(n);
      const L = new Float32Array(n);
      const W = new Float32Array(n);
      const dPost = new Float32Array(n);
      const psi = new Float32Array(n);
      const mode = new Int8Array(n);
      const brake = new Uint8Array(n);

      for (let i = 0; i < n; i++) {
        const veh = vs[i];
        const lenM = veh.L || 4.5;
        const s = ensureSlot(veh.id, veh.vtype, lenM);
        slot[i] = s;
        x[i] = veh.x;
        y[i] = veh.y;
        yaw[i] = veh.yaw;
        v[i] = veh.v;
        L[i] = lenM;
        W[i] = veh.W || 1.85;
        dPost[i] = veh.d_post || 0.0;
        psi[i] = veh.psi || 0.0;
        mode[i] = veh.mode !== undefined ? veh.mode : -1;
        brake[i] = veh.brake ? 1 : 0;
      }

      const rawEdges = f.dbn_edges ? f.dbn_edges.slice() : [];
      if (rawEdges.length > 1) {
        rawEdges.sort((a, b) => (b.omega || 0.0) - (a.omega || 0.0));
      }
      const m = rawEdges.length;
      const eSrc = new Int32Array(m);
      const eDst = new Int32Array(m);
      const eRel = new Uint8Array(m);
      const eOmega = new Float32Array(m);
      const eGamma = new Float32Array(m);
      let validEdges = 0;
      for (let e = 0; e < m; e++) {
        const ed = rawEdges[e];
        const sSlot = idToSlot.get(ed.src);
        const dSlot = idToSlot.get(ed.dst);
        if (sSlot === undefined || dSlot === undefined) continue;
        eSrc[validEdges] = sSlot;
        eDst[validEdges] = dSlot;
        eRel[validEdges] = Math.max(0, Math.min(3, ed.rel | 0));
        eOmega[validEdges] = ed.omega || 0.0;
        eGamma[validEdges] = ed.gamma !== undefined ? ed.gamma : ed.omega || 0.5;
        validEdges++;
      }

      // Flatten corridor spillback entries to avoid Object.entries() allocation per frame
      const heatArr = [];
      if (f.corridor_heat) {
        const hEntries = Object.entries(f.corridor_heat);
        for (let h = 0; h < hEntries.length; h++) {
          const cid = hEntries[h][0];
          const hv = hEntries[h][1];
          const spill = Math.max(hv.spill || 0.0, (hv.tail || 0.0) * 0.92);
          if (spill > 0.35) {
            heatArr.push({ cid, spill });
          }
        }
      }

      indexedFrames[fIdx] = {
        raw: f,
        t: f.t,
        n,
        slot,
        x,
        y,
        yaw,
        v,
        L,
        W,
        dPost,
        psi,
        mode,
        brake,
        edgeCount: validEdges,
        eSrc,
        eDst,
        eRel,
        eOmega,
        eGamma,
        heatArr,
        tls: f.tls || {},
        kpi: f.kpi || null,
      };
    }

    const tStart = indexedFrames.length > 0 ? indexedFrames[0].t : 0.0;
    const tEnd = indexedFrames.length > 0 ? indexedFrames[indexedFrames.length - 1].t : 0.0;
    const dt = trace.dt || (indexedFrames.length >= 2 ? Math.max(0.05, indexedFrames[1].t - indexedFrames[0].t) : 0.5);

    return {
      raw: trace,
      frames: indexedFrames,
      numSlots: Math.max(1, slotIds.length),
      idToSlot,
      slotIds,
      slotVtypes,
      slotIsVan: new Uint8Array(slotIsVan),
      tStart,
      tEnd,
      duration: Math.max(0.0, tEnd - tStart),
      dt,
    };
  }

  // ===========================================================================
  // 8. Cinema-Grade WebGL2 Viewport (Zero Per-Frame Heap Allocations)
  // ===========================================================================

  class CityWebGL2Viewport {
    constructor(canvas, scene, staticBuffers, indexedTrace, sharedVehicleMesh, sharedDecalQuads) {
      this.canvas = canvas;
      this.scene = scene;
      this.staticBuffers = staticBuffers;
      this.idxTrace = indexedTrace;

      const gl = canvas.getContext("webgl2", {
        antialias: true,
        alpha: false,
        depth: true,
        stencil: false,
        powerPreference: "high-performance",
        preserveDrawingBuffer: true,
      });
      if (!gl) {
        throw new Error("WebGL2 context creation failed. Please enable hardware acceleration in your browser.");
      }
      this.gl = gl;

      // Track CSS box dimensions via ResizeObserver (zero layout thrashing inside rAF)
      this._cssW = 0;
      this._cssH = 0;
      this._sizeDirty = true;
      if (typeof ResizeObserver === "function") {
        this._ro = new ResizeObserver((entries) => {
          for (let i = 0; i < entries.length; i++) {
            const cr = entries[i].contentRect;
            this._cssW = cr.width;
            this._cssH = cr.height;
            this._sizeDirty = false;
          }
        });
        this._ro.observe(canvas);
      }

      // Compile Shader Programs
      this.litProg = createProgram(gl, LIT_VS, LIT_FS);
      this.litU = {
        mvp: gl.getUniformLocation(this.litProg, "u_mvp"),
        eyePos: gl.getUniformLocation(this.litProg, "u_eyePos"),
        focusTarget: gl.getUniformLocation(this.litProg, "u_focusTarget"),
        sceneCenter: gl.getUniformLocation(this.litProg, "u_sceneCenter"),
        fogScale: gl.getUniformLocation(this.litProg, "u_fogScale"),
        occludeRadius: gl.getUniformLocation(this.litProg, "u_occludeRadius"),
        passMode: gl.getUniformLocation(this.litProg, "u_passMode"),
      };

      this.vehProg = createProgram(gl, VEH_VS, VEH_FS);
      this.vehU = {
        mvp: gl.getUniformLocation(this.vehProg, "u_mvp"),
        eyePos: gl.getUniformLocation(this.vehProg, "u_eyePos"),
      };

      this.decalProg = createProgram(gl, DECAL_VS, DECAL_FS);
      this.decalU = {
        mvp: gl.getUniformLocation(this.decalProg, "u_mvp"),
      };

      this.arcProg = createProgram(gl, ARC_VS, ARC_FS);
      this.arcU = {
        mvp: gl.getUniformLocation(this.arcProg, "u_mvp"),
        animTime: gl.getUniformLocation(this.arcProg, "u_animTime"),
      };

      // Static Geometry VAOs
      this.roadVao = this._createStaticLitVao(staticBuffers.roadFloats);
      this.roadVerts = staticBuffers.roadVertexCount;
      this.bldgVao = this._createStaticLitVao(staticBuffers.buildingFloats);
      this.bldgVerts = staticBuffers.buildingVertexCount;

      // Dynamic Signal Stop-Bars VAO
      this.sigLitBuilder = new LitMeshBuilder(32768);
      const sigStream = this._createStreamLitVao(32768);
      this.sigVao = sigStream.vao;
      this.sigVbo = sigStream.vbo;
      this.sigVboCapFloats = 32768;

      // Dynamic Luminous Arcs & Asphalt Light Pools VAO
      this.arcBuilder = new ArcMeshBuilder(98304);
      const arcStream = this._createStreamArcVao(98304);
      this.arcVao = arcStream.vao;
      this.arcVbo = arcStream.vbo;
      this.arcVboCapFloats = 98304;

      // GPU-Instanced Vehicle + Decal VAOs sharing one pre-allocated Float32Array instance buffer
      this.maxInstances = Math.max(512, indexedTrace.numSlots + 32);
      this.vehInstanceData = new Float32Array(this.maxInstances * 12);
      this.vehTemplateVerts = sharedVehicleMesh.vertexCount;

      this.vehInstanceVbo = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, this.vehInstanceVbo);
      gl.bufferData(gl.ARRAY_BUFFER, this.vehInstanceData.byteLength, gl.DYNAMIC_DRAW);

      this.vehVao = this._createInstancedVehicleVao(sharedVehicleMesh.data, this.vehInstanceVbo);
      this.decalVao = this._createInstancedDecalVao(sharedDecalQuads, this.vehInstanceVbo);

      // Zero-GC Sub-Tick Interpolation Slot Buffers
      const V = indexedTrace.numSlots;
      this.slotX = new Float32Array(V);
      this.slotY = new Float32Array(V);
      this.slotYaw = new Float32Array(V);
      this.slotV = new Float32Array(V);
      this.slotDPost = new Float32Array(V);
      this.slotPsi = new Float32Array(V);
      this.slotMode = new Int8Array(V);
      this.slotActiveGen = new Int32Array(V);
      this._activeGen = 1;

      this._f1SlotGen = new Int32Array(V);
      this._f1SlotIdx = new Int32Array(V);
      this._f1Gen = 1;
      this._cachedIdx1 = -1;
      this._srcArcGen = new Int32Array(V);
      this._srcArcCount = new Uint8Array(V);
      this._arcPassGen = 1;
      this._edgeDrawnMask = new Uint8Array(256);

      this.activeInstanceCount = 0;
      this.currentFrameObj = null;
      this.currentSimTime = indexedTrace.tStart;

      // Pre-allocated camera matrices (zero mat4 allocations per frame)
      this._projMat = mat4Identity();
      this._viewMat = mat4Identity();
      this.lastMvp = mat4Identity();
      this._eyeVec = new Float32Array(3);
      this._targetVec = new Float32Array(3);
    }

    _createStaticLitVao(floats) {
      const gl = this.gl;
      const vao = gl.createVertexArray();
      const vbo = gl.createBuffer();
      gl.bindVertexArray(vao);
      gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
      gl.bufferData(gl.ARRAY_BUFFER, floats, gl.STATIC_DRAW);
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

    _createStreamLitVao(capFloats) {
      const gl = this.gl;
      const vao = gl.createVertexArray();
      const vbo = gl.createBuffer();
      gl.bindVertexArray(vao);
      gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
      gl.bufferData(gl.ARRAY_BUFFER, capFloats * 4, gl.DYNAMIC_DRAW);
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

    _createStreamArcVao(capFloats) {
      const gl = this.gl;
      const vao = gl.createVertexArray();
      const vbo = gl.createBuffer();
      gl.bindVertexArray(vao);
      gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
      gl.bufferData(gl.ARRAY_BUFFER, capFloats * 4, gl.DYNAMIC_DRAW);
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

    _createInstancedVehicleVao(templateFloats, instanceVbo) {
      const gl = this.gl;
      const vao = gl.createVertexArray();
      const tplVbo = gl.createBuffer();
      gl.bindVertexArray(vao);

      gl.bindBuffer(gl.ARRAY_BUFFER, tplVbo);
      gl.bufferData(gl.ARRAY_BUFFER, templateFloats, gl.STATIC_DRAW);
      const tplStride = 7 * 4;
      gl.enableVertexAttribArray(0);
      gl.vertexAttribPointer(0, 3, gl.FLOAT, false, tplStride, 0);
      gl.enableVertexAttribArray(1);
      gl.vertexAttribPointer(1, 3, gl.FLOAT, false, tplStride, 3 * 4);
      gl.enableVertexAttribArray(2);
      gl.vertexAttribPointer(2, 1, gl.FLOAT, false, tplStride, 6 * 4);

      gl.bindBuffer(gl.ARRAY_BUFFER, instanceVbo);
      const instStride = 12 * 4;
      gl.enableVertexAttribArray(3);
      gl.vertexAttribPointer(3, 4, gl.FLOAT, false, instStride, 0);
      gl.vertexAttribDivisor(3, 1);
      gl.enableVertexAttribArray(4);
      gl.vertexAttribPointer(4, 4, gl.FLOAT, false, instStride, 4 * 4);
      gl.vertexAttribDivisor(4, 1);
      gl.enableVertexAttribArray(5);
      gl.vertexAttribPointer(5, 4, gl.FLOAT, false, instStride, 8 * 4);
      gl.vertexAttribDivisor(5, 1);

      gl.bindVertexArray(null);
      return vao;
    }

    _createInstancedDecalVao(decalQuadFloats, instanceVbo) {
      const gl = this.gl;
      const vao = gl.createVertexArray();
      const quadVbo = gl.createBuffer();
      gl.bindVertexArray(vao);

      gl.bindBuffer(gl.ARRAY_BUFFER, quadVbo);
      gl.bufferData(gl.ARRAY_BUFFER, decalQuadFloats, gl.STATIC_DRAW);
      gl.enableVertexAttribArray(0);
      gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 3 * 4, 0);

      gl.bindBuffer(gl.ARRAY_BUFFER, instanceVbo);
      const instStride = 12 * 4;
      gl.enableVertexAttribArray(3);
      gl.vertexAttribPointer(3, 4, gl.FLOAT, false, instStride, 0);
      gl.vertexAttribDivisor(3, 1);
      gl.enableVertexAttribArray(4);
      gl.vertexAttribPointer(4, 4, gl.FLOAT, false, instStride, 4 * 4);
      gl.vertexAttribDivisor(4, 1);
      gl.enableVertexAttribArray(5);
      gl.vertexAttribPointer(5, 4, gl.FLOAT, false, instStride, 8 * 4);
      gl.vertexAttribDivisor(5, 1);

      gl.bindVertexArray(null);
      return vao;
    }

    syncCanvasSize() {
      if (this._sizeDirty || (this._cssW < 2 && this._cssH < 2 && typeof ResizeObserver !== "function")) {
        this._cssW = this.canvas.clientWidth || 0;
        this._cssH = this.canvas.clientHeight || 0;
        this._sizeDirty = false;
      }
      const dpr = Math.min(2.0, window.devicePixelRatio || 1.0);
      const w = Math.floor(this._cssW * dpr);
      const h = Math.floor(this._cssH * dpr);
      if (w < 2 || h < 2) {
        return { width: w, height: h, aspect: 1.0, valid: false };
      }
      if (this.canvas.width !== w || this.canvas.height !== h) {
        this.canvas.width = w;
        this.canvas.height = h;
      }
      return { width: w, height: h, aspect: w / h, valid: true };
    }

    interpolateSubTickVehicles(simTime, selectedSlot) {
      const tr = this.idxTrace;
      const frames = tr.frames;
      const nFrames = frames.length;
      if (nFrames === 0) {
        this.activeInstanceCount = 0;
        this.currentFrameObj = null;
        return;
      }

      const clampedSim = Math.max(tr.tStart, Math.min(tr.tEnd, simTime));
      this.currentSimTime = clampedSim;

      // Fast O(1) uniform-dt lookup with monotonicity adjustment
      let idx0 = Math.min(nFrames - 1, Math.max(0, Math.floor((clampedSim - tr.tStart) / tr.dt)));
      while (idx0 < nFrames - 1 && frames[idx0 + 1].t <= clampedSim) idx0++;
      while (idx0 > 0 && frames[idx0].t > clampedSim) idx0--;
      const idx1 = Math.min(nFrames - 1, idx0 + 1);

      const f0 = frames[idx0];
      const f1 = frames[idx1];
      const spanT = f1.t - f0.t;
      const alpha = spanT > 1e-6 ? Math.max(0.0, Math.min(1.0, (clampedSim - f0.t) / spanT)) : 0.0;

      // Refresh frame[idx1] slot index lookup only when idx1 advances (at 2 Hz)
      if (this._cachedIdx1 !== idx1) {
        const g1 = ++this._f1Gen;
        const s1 = f1.slot;
        for (let j = 0; j < f1.n; j++) {
          const sl = s1[j];
          this._f1SlotGen[sl] = g1;
          this._f1SlotIdx[sl] = j;
        }
        this._cachedIdx1 = idx1;
      }

      const actGen = ++this._activeGen;
      const f1Gen = this._f1Gen;
      const inst = this.vehInstanceData;
      const n0 = f0.n;

      // Cubic Hermite basis weights for smooth intersection turning arcs
      const a2 = alpha * alpha;
      const a3 = a2 * alpha;
      const h00 = 2.0 * a3 - 3.0 * a2 + 1.0;
      const h10 = a3 - 2.0 * a2 + alpha;
      const h01 = -2.0 * a3 + 3.0 * a2;
      const h11 = a3 - a2;

      let outCount = 0;
      for (let i = 0; i < n0; i++) {
        if (outCount >= this.maxInstances) break;
        const s = f0.slot[i];
        let ix = f0.x[i];
        let iy = f0.y[i];
        let iyaw = f0.yaw[i];
        let iv = f0.v[i];
        let idPost = f0.dPost[i];
        let ipsi = f0.psi[i];

        if (idx1 !== idx0 && this._f1SlotGen[s] === f1Gen) {
          const j = this._f1SlotIdx[s];
          const x1 = f1.x[j], y1 = f1.y[j];
          const dx = x1 - ix, dy = y1 - iy;
          const chord = Math.hypot(dx, dy);
          if (chord < 40.0) {
            const yaw1 = f1.yaw[j];
            const v1 = f1.v[j];
            const dYaw = Math.atan2(Math.sin(yaw1 - iyaw), Math.cos(yaw1 - iyaw));

            if (Math.abs(dYaw) > 0.06 && chord > 0.35 && iv > 0.8 && v1 > 0.8) {
              // Curvature-clamped Hermite arc so turning vehicles round corners smoothly without overshoot
              const t0Mag = Math.min(iv * spanT, chord * 1.05);
              const t1Mag = Math.min(v1 * spanT, chord * 1.05);
              ix = h00 * ix + h10 * Math.cos(iyaw) * t0Mag + h01 * x1 + h11 * Math.cos(yaw1) * t1Mag;
              iy = h00 * iy + h10 * Math.sin(iyaw) * t0Mag + h01 * y1 + h11 * Math.sin(yaw1) * t1Mag;
            } else {
              ix += dx * alpha;
              iy += dy * alpha;
            }
            iyaw = iyaw + dYaw * alpha;
            iv = iv + (v1 - iv) * alpha;
            idPost = idPost + (f1.dPost[j] - idPost) * alpha;
            ipsi = ipsi + (f1.psi[j] - ipsi) * alpha;
          }
        }

        const modeInt = alpha < 0.5 ? f0.mode[i] : (idx1 !== idx0 && this._f1SlotGen[s] === f1Gen ? f1.mode[this._f1SlotIdx[s]] : f0.mode[i]);
        const isBraking = f0.brake[i] !== 0 || iv < 0.6;
        const isVan = tr.slotIsVan[s];
        const isSelected = s === selectedSlot ? 1 : 0;

        this.slotX[s] = ix;
        this.slotY[s] = iy;
        this.slotYaw[s] = iyaw;
        this.slotV[s] = iv;
        this.slotDPost[s] = idPost;
        this.slotPsi[s] = ipsi;
        this.slotMode[s] = modeInt;
        this.slotActiveGen[s] = actGen;

        const base = outCount * 12;
        inst[base + 0] = ix;
        inst[base + 1] = iy;
        inst[base + 2] = iyaw;
        inst[base + 3] = iv;

        inst[base + 4] = f0.L[i];
        inst[base + 5] = f0.W[i];
        inst[base + 6] = isVan ? 2.22 : 1.52;
        inst[base + 7] = (isVan ? 1 : 0) | (isSelected ? 2 : 0);

        writeVehicleColorRGB(inst, base + 8, iv, idPost, modeInt);
        inst[base + 11] = isBraking ? 1.0 : 0.24;

        outCount++;
      }

      this.activeInstanceCount = outCount;
      this.currentFrameObj = alpha < 0.5 ? f0 : f1;
    }

    buildSignalAndOverlayMeshes(eyePos, targetPos, cameraDistance, opts, selectedSlot) {
      const sigLit = this.sigLitBuilder;
      const arc = this.arcBuilder;
      sigLit.clear();
      arc.clear();

      const frame = this.currentFrameObj;
      if (!frame) return;

      // 1. Vivid 3D Glowing Traffic Signal Stop-Bars + Overhead Signal Lenses + Soft Asphalt Light Pools
      const tlsStates = frame.tls;
      const sigDescs = this.staticBuffers.signalLaneDescriptors;
      for (let i = 0; i < sigDescs.length; i++) {
        const d = sigDescs[i];
        const stateStr = tlsStates[d.tlsId] || "";
        let hasGreen = false;
        let hasYellow = false;
        const idxs = d.linkIndices;
        for (let k = 0; k < idxs.length; k++) {
          const ch = stateStr.charAt(idxs[k]);
          if (ch === "G" || ch === "g") {
            hasGreen = true;
            break;
          } else if (ch === "y" || ch === "Y") {
            hasYellow = true;
          }
        }

        // #10b981 Emerald Green, #f59e0b Amber, #ef4444 Ruby Red
        let r = 0.937, g = 0.267, b = 0.267;
        let poolAlpha = 0.24;
        if (hasGreen) {
          r = 0.063; g = 0.785; b = 0.535;
          poolAlpha = 0.26;
        } else if (hasYellow) {
          r = 0.961; g = 0.620; b = 0.043;
          poolAlpha = 0.25;
        }

        const bulbCol = [r, g, b, 1.0];
        // Raised 3D glowing signal stop-bar across the incoming lane
        sigLit.pushOrientedBox(
          d.cx,
          d.cy,
          0.075,
          0.165,
          d.travelYaw,
          0.28,
          0.46 * d.widthM,
          bulbCol,
          3.0,
          0.92
        );
        // Overhead glowing signal lens
        sigLit.pushOrientedBox(
          d.cx,
          d.cy,
          d.headZ - 0.18,
          d.headZ + 0.18,
          d.faceYaw,
          0.22,
          0.20,
          bulbCol,
          3.0,
          1.0
        );
        // Soft colored light pool bathing the asphalt at the intersection approach
        const poolShiftX = d.cx + Math.cos(d.travelYaw) * 1.3;
        const poolShiftY = d.cy + Math.sin(d.travelYaw) * 1.3;
        arc.pushLightPoolDisc(poolShiftX, poolShiftY, 0.078, Math.max(3.6, d.widthM * 1.35), [r, g, b, poolAlpha]);
      }

      // 2. Subtle Warm Amber/Crimson Corridor Border Glow when spill > 0.35
      const heatArr = frame.heatArr;
      const borders = this.staticBuffers.corridorBorders;
      for (let h = 0; h < heatArr.length; h++) {
        const item = heatArr[h];
        const shapes = borders.get(item.cid);
        if (!shapes) continue;
        const t = Math.min(1.0, (item.spill - 0.35) / 0.55);
        const hCol = [0.97, 0.62 - 0.38 * t, 0.14 + 0.12 * t, 0.14 + 0.28 * t];
        for (let s = 0; s < shapes.length; s++) {
          const pts = shapes[s];
          const sideSign = s === 0 ? -1.0 : 1.0;
          for (let k = 0; k < pts.length - 1; k++) {
            const p0 = pts[k], p1 = pts[k + 1];
            const dx = p1[0] - p0[0], dy = p1[1] - p0[1];
            const slen = Math.hypot(dx, dy);
            if (slen < 0.2) continue;
            const ux = -dy / slen, uy = dx / slen;
            const off = sideSign * 2.05;
            const hw = 0.55;
            const cx0 = p0[0] + ux * off, cy0 = p0[1] + uy * off;
            const cx1 = p1[0] + ux * off, cy1 = p1[1] + uy * off;
            arc.pushQuadUV(
              [cx0 - ux * hw, cy0 - uy * hw, 0.085],
              [cx1 - ux * hw, cy1 - uy * hw, 0.085],
              [cx1 + ux * hw, cy1 + uy * hw, 0.085],
              [cx0 + ux * hw, cy0 + uy * hw, 0.085],
              0.0,
              1.0,
              hCol,
              item.spill,
              -1.0
            );
          }
        }
      }

      // 3. Selected / Chased Vehicle Crisp Cyan-Gold Target Halo
      const actGen = this._activeGen;
      if (selectedSlot >= 0 && this.slotActiveGen[selectedSlot] === actGen) {
        arc.pushRing2D(
          this.slotX[selectedSlot],
          this.slotY[selectedSlot],
          0.12,
          3.1,
          3.85,
          28,
          [0.22, 0.90, 0.98, 0.78]
        );
      }

      // 4. Sleek Elevated 3D Parabolic Causal DBN Arcs (omega >= 0.18, spatially balanced & distance-scaled)
      if (opts.showDBN && frame.edgeCount > 0) {
        const maxArcs = 56;
        const eCount = frame.edgeCount;
        if (this._edgeDrawnMask.length < eCount) {
          this._edgeDrawnMask = new Uint8Array(Math.max(256, eCount * 2));
        }
        const drawnMask = this._edgeDrawnMask;
        for (let e = 0; e < eCount; e++) drawnMask[e] = 0;

        const arcGen = ++this._arcPassGen;
        const srcGen = this._srcArcGen;
        const srcCnt = this._srcArcCount;
        const distScale = Math.max(1.0, Math.min(3.3, cameraDistance / 125.0));
        const viewRadius = Math.max(180.0, cameraDistance * 0.95);
        const tx = targetPos[0], ty = targetPos[1];

        let drawn = 0;
        const tryEmitArc = (e, requireInView, maxPerSrc) => {
          if (drawnMask[e]) return;
          const omega = frame.eOmega[e];
          const sSlot = frame.eSrc[e];
          const dSlot = frame.eDst[e];
          const touchesSelected = selectedSlot >= 0 && (sSlot === selectedSlot || dSlot === selectedSlot);
          if (omega < 0.18 && !(touchesSelected && omega >= 0.10)) return;
          if (this.slotActiveGen[sSlot] !== actGen || this.slotActiveGen[dSlot] !== actGen) return;

          if (srcGen[sSlot] !== arcGen) {
            srcGen[sSlot] = arcGen;
            srcCnt[sSlot] = 0;
          }
          if (!touchesSelected && srcCnt[sSlot] >= maxPerSrc) return;

          const x0 = this.slotX[sSlot], y0 = this.slotY[sSlot];
          const x1 = this.slotX[dSlot], y1 = this.slotY[dSlot];
          if (requireInView && !touchesSelected) {
            const d0 = Math.hypot(x0 - tx, y0 - ty);
            const d1 = Math.hypot(x1 - tx, y1 - ty);
            if (d0 > viewRadius && d1 > viewRadius) return;
          }

          const span = Math.hypot(x1 - x0, y1 - y0);
          if (span < 2.2 || span > 260.0) return;

          const relIdx = frame.eRel[e];
          const arcHeight = Math.min(28.0, 5.2 + 0.19 * span + 1.6 * relIdx);
          const halfW = distScale * (0.24 + 0.30 * omega) * (touchesSelected ? 1.30 : 1.0);
          const rc = REL_COLORS[relIdx];
          const alpha = Math.min(0.95, (0.48 + 0.46 * omega) * (touchesSelected ? 1.15 : 1.0));

          arc.pushParabolicArc3D(
            x0,
            y0,
            x1,
            y1,
            1.85,
            arcHeight,
            halfW,
            [rc[0], rc[1], rc[2], alpha],
            omega,
            frame.eGamma[e],
            eyePos
          );
          drawnMask[e] = 1;
          srcCnt[sSlot]++;
          drawn++;
        };

        // Pass A: Selected vehicle causal neighborhood
        if (selectedSlot >= 0) {
          for (let e = 0; e < eCount && drawn < maxArcs; e++) {
            if (frame.eSrc[e] === selectedSlot || frame.eDst[e] === selectedSlot) {
              tryEmitArc(e, false, 8);
            }
          }
        }
        // Pass B: High-confidence diverse edges inside the active viewport focus
        for (let e = 0; e < eCount && drawn < maxArcs; e++) {
          tryEmitArc(e, true, 2);
        }
        // Pass C: Fill remaining budget across wider scene
        for (let e = 0; e < eCount && drawn < maxArcs; e++) {
          tryEmitArc(e, false, 3);
        }
      }
    }

    render(eye, target, cameraDistance, isOrtho, enableBuildingFade, simTime, wallSec, opts, selectedSlot) {
      const sz = this.syncCanvasSize();
      if (!sz.valid) return false;

      const gl = this.gl;
      gl.viewport(0, 0, sz.width, sz.height);
      // Deep obsidian-navy clear (#080c14)
      gl.clearColor(0.031, 0.047, 0.078, 1.0);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      gl.enable(gl.DEPTH_TEST);
      gl.depthFunc(gl.LEQUAL);

      this.interpolateSubTickVehicles(simTime, selectedSlot);

      if (isOrtho) {
        mat4Ortho(this._projMat, cameraDistance * 0.52, sz.aspect, -1800.0, 3800.0);
        mat4LookAtZUp(this._viewMat, eye, target, 0.0, 1.0, 0.0);
      } else {
        mat4Perspective(this._projMat, (44.0 * Math.PI) / 180.0, sz.aspect, 1.5, 4800.0);
        mat4LookAtZUp(this._viewMat, eye, target, 0.0, 0.0, 1.0);
      }
      mat4Multiply(this.lastMvp, this._projMat, this._viewMat);

      this.buildSignalAndOverlayMeshes(eye, target, cameraDistance, opts, selectedSlot);

      const b = this.scene.bounds;
      const fogScale = Math.max(320.0, 0.90 * Math.hypot(b.span_x, b.span_y));
      const occludeRadius = enableBuildingFade ? Math.max(12.0, Math.min(22.0, cameraDistance * 0.14)) : 0.0;

      // Pass 1: Opaque Urban Scene (Ground, Parks, Roads, Signal Stop-Bars, 3D Buildings)
      gl.disable(gl.BLEND);
      gl.disable(gl.CULL_FACE);
      gl.depthMask(true);
      gl.useProgram(this.litProg);
      gl.uniformMatrix4fv(this.litU.mvp, false, this.lastMvp);
      gl.uniform3f(this.litU.eyePos, eye[0], eye[1], eye[2]);
      gl.uniform3f(this.litU.focusTarget, target[0], target[1], target[2]);
      gl.uniform3f(this.litU.sceneCenter, b.center_x, b.center_y, 0.0);
      gl.uniform1f(this.litU.fogScale, fogScale);
      gl.uniform1f(this.litU.occludeRadius, occludeRadius);
      gl.uniform1i(this.litU.passMode, 0);

      gl.bindVertexArray(this.roadVao);
      gl.drawArrays(gl.TRIANGLES, 0, this.roadVerts);

      if (opts.showBuildings && this.bldgVerts > 0) {
        gl.bindVertexArray(this.bldgVao);
        gl.drawArrays(gl.TRIANGLES, 0, this.bldgVerts);
      }

      const sigVerts = this.sigLitBuilder.vertexCount;
      if (sigVerts > 0) {
        gl.bindVertexArray(this.sigVao);
        gl.bindBuffer(gl.ARRAY_BUFFER, this.sigVbo);
        if (this.sigLitBuilder.len > this.sigVboCapFloats) {
          this.sigVboCapFloats = this.sigLitBuilder.data.length;
          gl.bufferData(gl.ARRAY_BUFFER, this.sigVboCapFloats * 4, gl.DYNAMIC_DRAW);
        }
        gl.bufferSubData(gl.ARRAY_BUFFER, 0, this.sigLitBuilder.data.subarray(0, this.sigLitBuilder.len));
        gl.drawArrays(gl.TRIANGLES, 0, sigVerts);
      }

      // Upload per-frame GPU vehicle instance buffer ONCE (shared by Ground Decals + 3D Vehicles)
      const nVeh = this.activeInstanceCount;
      if (nVeh > 0) {
        gl.bindBuffer(gl.ARRAY_BUFFER, this.vehInstanceVbo);
        gl.bufferSubData(gl.ARRAY_BUFFER, 0, this.vehInstanceData.subarray(0, nVeh * 12));

        // Pass 1B: Ground Contact Shadows + Headlight Cones + Ruby Brake Pools on Asphalt
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
        gl.depthMask(false);
        gl.useProgram(this.decalProg);
        gl.uniformMatrix4fv(this.decalU.mvp, false, this.lastMvp);
        gl.bindVertexArray(this.decalVao);
        gl.drawArraysInstanced(gl.TRIANGLES, 0, 18, nVeh);

        // Pass 1C: Opaque GPU-Instanced Sculpted 3D Vehicles
        gl.disable(gl.BLEND);
        gl.depthMask(true);
        gl.useProgram(this.vehProg);
        gl.uniformMatrix4fv(this.vehU.mvp, false, this.lastMvp);
        gl.uniform3f(this.vehU.eyePos, eye[0], eye[1], eye[2]);
        gl.bindVertexArray(this.vehVao);
        gl.drawArraysInstanced(gl.TRIANGLES, 0, this.vehTemplateVerts, nVeh);
      }

      // Pass 2: Ghosted X-Ray Shell for Tall Buildings Blocking the Camera Ray
      if (opts.showBuildings && occludeRadius > 0.5 && this.bldgVerts > 0) {
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
        gl.enable(gl.CULL_FACE);
        gl.depthMask(false);
        gl.useProgram(this.litProg);
        gl.uniform1i(this.litU.passMode, 1);
        gl.bindVertexArray(this.bldgVao);
        gl.drawArrays(gl.TRIANGLES, 0, this.bldgVerts);
        gl.disable(gl.CULL_FACE);
      }

      // Pass 3: Luminous Additive 3D Parabolic Causal DBN Arcs & Signal Light Pools
      const arcVerts = this.arcBuilder.vertexCount;
      if (arcVerts > 0) {
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE);
        gl.depthMask(false);
        gl.useProgram(this.arcProg);
        gl.uniformMatrix4fv(this.arcU.mvp, false, this.lastMvp);
        gl.uniform1f(this.arcU.animTime, wallSec);

        gl.bindVertexArray(this.arcVao);
        gl.bindBuffer(gl.ARRAY_BUFFER, this.arcVbo);
        if (this.arcBuilder.len > this.arcVboCapFloats) {
          this.arcVboCapFloats = this.arcBuilder.data.length;
          gl.bufferData(gl.ARRAY_BUFFER, this.arcVboCapFloats * 4, gl.DYNAMIC_DRAW);
        }
        gl.bufferSubData(gl.ARRAY_BUFFER, 0, this.arcBuilder.data.subarray(0, this.arcBuilder.len));
        gl.drawArrays(gl.TRIANGLES, 0, arcVerts);
      }

      gl.depthMask(true);
      gl.disable(gl.BLEND);
      gl.bindVertexArray(null);
      return true;
    }

    pickVehicleAtClientXY(clientX, clientY) {
      const rect = this.canvas.getBoundingClientRect();
      if (rect.width < 2 || rect.height < 2) return null;
      const dpr = this.canvas.width / rect.width;
      const sx = (clientX - rect.left) * dpr;
      const sy = (clientY - rect.top) * dpr;

      const mvp = this.lastMvp;
      const w = this.canvas.width;
      const h = this.canvas.height;
      const actGen = this._activeGen;
      const tr = this.idxTrace;

      let bestId = null;
      let bestDistPx = 32.0 * dpr;
      const nSlots = tr.numSlots;
      for (let s = 0; s < nSlots; s++) {
        if (this.slotActiveGen[s] !== actGen) continue;
        const x = this.slotX[s], y = this.slotY[s], z = 1.0;
        const cw = mvp[3] * x + mvp[7] * y + mvp[11] * z + mvp[15];
        if (cw <= 0.1) continue;
        const cx = mvp[0] * x + mvp[4] * y + mvp[8] * z + mvp[12];
        const cy = mvp[1] * x + mvp[5] * y + mvp[9] * z + mvp[13];
        const px = ((cx / cw) * 0.5 + 0.5) * w;
        const py = (0.5 - (cy / cw) * 0.5) * h;
        const dist = Math.hypot(px - sx, py - sy);
        if (dist < bestDistPx) {
          bestDistPx = dist;
          bestId = tr.slotIds[s];
        }
      }
      return bestId;
    }

    findBestLeadVehicleId(focusX, focusY, peerViewport) {
      const actGen = this._activeGen;
      const tr = this.idxTrace;
      const frame = this.currentFrameObj;
      const edgeTouchGen = ++this._arcPassGen;
      const srcGen = this._srcArcGen;
      const srcCnt = this._srcArcCount;
      if (frame && frame.edgeCount > 0) {
        for (let e = 0; e < frame.edgeCount; e++) {
          if (frame.eOmega[e] < 0.18) continue;
          const s0 = frame.eSrc[e];
          const s1 = frame.eDst[e];
          if (srcGen[s0] !== edgeTouchGen) { srcGen[s0] = edgeTouchGen; srcCnt[s0] = 0; }
          if (srcGen[s1] !== edgeTouchGen) { srcGen[s1] = edgeTouchGen; srcCnt[s1] = 0; }
          if (srcCnt[s0] < 10) srcCnt[s0]++;
          if (srcCnt[s1] < 10) srcCnt[s1]++;
        }
      }

      let bestId = null;
      let bestScore = -1e9;
      for (let s = 0; s < tr.numSlots; s++) {
        if (this.slotActiveGen[s] !== actGen) continue;
        const x = this.slotX[s], y = this.slotY[s];
        const v = this.slotV[s];
        const distCenter = Math.hypot(x - focusX, y - focusY);

        // Strongly prefer moving vehicles inside the urban signalized core
        const speedTerm = v >= 2.5 ? 6.5 + Math.min(4.0, v * 0.42) : (v >= 1.2 ? 1.5 : -8.5);
        const coreTerm = distCenter <= 220.0 ? 4.8 * (1.0 - distCenter / 220.0) : -Math.min(9.0, (distCenter - 220.0) * 0.035);
        const deg = srcGen[s] === edgeTouchGen ? srcCnt[s] : 0;
        const dbnTerm = Math.min(4.5, deg * 1.35);

        // In split-screen, favor locations where comparison viewport also has active nearby vehicles
        let peerTerm = 0.0;
        if (peerViewport) {
          const pIdSlot = peerViewport.idxTrace.idToSlot.get(tr.slotIds[s]);
          if (pIdSlot !== undefined && peerViewport.slotActiveGen[pIdSlot] === peerViewport._activeGen) {
            const pd = Math.hypot(peerViewport.slotX[pIdSlot] - x, peerViewport.slotY[pIdSlot] - y);
            peerTerm += pd < 48.0 ? 4.0 : 1.2;
          }
        }

        const score = speedTerm + coreTerm + dbnTerm + peerTerm + this.slotPsi[s] * 2.2;
        if (score > bestScore) {
          bestScore = score;
          bestId = tr.slotIds[s];
        }
      }
      return bestId;
    }

    getVehicleSnapshotById(vehId) {
      if (!vehId) return null;
      const slot = this.idxTrace.idToSlot.get(vehId);
      if (slot === undefined || this.slotActiveGen[slot] !== this._activeGen) return null;
      return {
        id: vehId,
        vtype: this.idxTrace.slotVtypes[slot],
        x: this.slotX[slot],
        y: this.slotY[slot],
        yaw: this.slotYaw[slot],
        v: this.slotV[slot],
        d_post: this.slotDPost[slot],
        psi: this.slotPsi[slot],
        mode: this.slotMode[slot],
      };
    }

    snapshotAllActiveVehicles() {
      const out = [];
      const actGen = this._activeGen;
      const tr = this.idxTrace;
      for (let s = 0; s < tr.numSlots; s++) {
        if (this.slotActiveGen[s] !== actGen) continue;
        out.push({
          id: tr.slotIds[s],
          x: this.slotX[s],
          y: this.slotY[s],
          yaw: this.slotYaw[s],
          v: this.slotV[s],
          d_post: this.slotDPost[s],
          psi: this.slotPsi[s],
          mode: this.slotMode[s],
        });
      }
      return out;
    }
  }

  // ===========================================================================
  // 9. Humanized Labels & Scorecard Formatting
  // ===========================================================================

  function formatScenarioTitle(scenarioId, networkId) {
    const key = String(scenarioId || networkId || "").toLowerCase();
    if (key.indexOf("palo_alto") >= 0) return "Downtown Palo Alto";
    if (key.indexOf("grid4x4_incident") >= 0) return "4×4 Incident Grid";
    if (key.indexOf("grid4x4") >= 0) return "4×4 Urban Grid";
    if (key.indexOf("grid3x3_noisy") >= 0) return "3×3 V2X Dropout Grid";
    if (key.indexOf("grid3x3") >= 0) return "3×3 Rush Grid";
    const raw = String(scenarioId || networkId || "Urban Network").replace(/_/g, " ");
    return raw.replace(/\b\w/g, (c) => c.toUpperCase());
  }

  function formatPolicyTitle(policyName, fallback) {
    const p = String(policyName || "").toLowerCase();
    if (p === "neural_dbn") return "Neural DBN Swarm";
    if (p === "actuated_signal") return "Actuated Signals";
    if (p === "fixed_signal") return "Fixed-Time Signals";
    if (p === "online_bic_dbn") return "Online BIC DBN";
    if (p === "heuristic_radius_dbn") return "Radius DBN";
    if (p === "independent_belief") return "Independent Belief";
    if (p === "fcfs_reservation") return "FCFS Reservation";
    if (!p) return fallback;
    return String(policyName).replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
  }

  function formatMMSS(sec) {
    const s = Math.max(0, Math.floor(sec + 1e-4));
    const mm = Math.floor(s / 60);
    const ss = s % 60;
    return (mm < 10 ? "0" + mm : String(mm)) + ":" + (ss < 10 ? "0" + ss : String(ss));
  }

  function formatSpillbackSec(val) {
    if (!Number.isFinite(val)) return "--";
    if (val < 100.0) return val.toFixed(1) + "s";
    return Math.round(val).toLocaleString("en-US") + "s";
  }

  function populateStaticScorecards(primaryTrace, compareTrace) {
    const setTxt = (id, txt) => {
      const el = document.getElementById(id);
      if (el) el.textContent = txt;
    };

    setTxt("title-primary", formatPolicyTitle(primaryTrace.policy_name, "Neural DBN Swarm"));
    if (compareTrace) {
      setTxt("title-compare", formatPolicyTitle(compareTrace.policy_name, "Actuated Signals"));
    }

    const mP = primaryTrace.summary_metrics;
    const mC = compareTrace ? compareTrace.summary_metrics : null;

    if (mP) {
      setTxt("kpi-p-delay", Number(mP.cohort_mean_total_delay_s).toFixed(1) + "s");
      setTxt("kpi-p-throughput", Math.round(Number(mP.throughput_veh_per_hr)).toLocaleString("en-US") + " /h");
      setTxt("kpi-p-spill", formatSpillbackSec(Number(mP.spillback_corridor_seconds)));
      const colsP = Number(mP.collision_count || 0);
      setTxt("kpi-p-safety", colsP + (colsP === 1 ? " Collision" : " Collisions"));
    }

    if (mC) {
      setTxt("kpi-c-delay", Number(mC.cohort_mean_total_delay_s).toFixed(1) + "s");
      setTxt("kpi-c-throughput", Math.round(Number(mC.throughput_veh_per_hr)).toLocaleString("en-US") + " /h");
      setTxt("kpi-c-spill", formatSpillbackSec(Number(mC.spillback_corridor_seconds)));
      const colsC = Number(mC.collision_count || 0);
      setTxt("kpi-c-safety", colsC + (colsC === 1 ? " Collision" : " Collisions"));
    }

    const badgeDelta = document.getElementById("badge-delta");
    if (badgeDelta && mP && mC) {
      const dP = Number(mP.cohort_mean_total_delay_s);
      const dC = Number(mC.cohort_mean_total_delay_s);
      const tP = Number(mP.throughput_veh_per_hr);
      const tC = Number(mC.throughput_veh_per_hr);
      const sP = Number(mP.spillback_corridor_seconds);
      const sC = Number(mC.spillback_corridor_seconds);

      const parts = [];
      if (dC > 0.1) {
        const dPct = ((dP - dC) / dC) * 100.0;
        parts.push((dPct <= 0 ? "−" + Math.abs(dPct).toFixed(1) : "+" + dPct.toFixed(1)) + "% Delay");
      }
      if (sC >= 20.0 && sP >= 0.1 && sC / sP >= 5.0) {
        parts.push(Math.round(sC / sP) + "× Less Spillback");
      } else if (tC > 1.0 && ((tP - tC) / tC) * 100.0 >= 3.0) {
        parts.push("+" + (((tP - tC) / tC) * 100.0).toFixed(1) + "% Throughput");
      } else if (sC > 5.0 && sP < sC) {
        parts.push("−" + (((sC - sP) / sC) * 100.0).toFixed(0) + "% Spillback");
      }

      if (parts.length > 0) {
        badgeDelta.textContent = parts.join(" · ");
        badgeDelta.hidden = false;
        badgeDelta.style.display = "";
      }
    }
  }

  // ===========================================================================
  // 10. Application Controller, Damped Camera & Minimalist HUD Binding
  // ===========================================================================

  function mountCityDBNViewerInstance(payload, bootStartMs) {
    const scene = payload.scene;
    const primaryTrace = payload.primary_trace;
    const compareTrace = payload.comparison_trace || null;
    const hasCompare = hasTraceFrames(compareTrace);

    const staticBuffers = buildStaticCityBuffers(scene);
    const sharedVehicleMesh = buildSculptedVehicleTemplate();
    const sharedDecalQuads = buildDecalQuadBuffer();

    const idxPrimary = buildIndexedTrace(primaryTrace);
    const idxCompare = hasCompare ? buildIndexedTrace(compareTrace) : null;

    const canvasPrimary = document.getElementById("canvas-primary");
    const canvasCompare = document.getElementById("canvas-compare");
    if (!canvasPrimary) {
      throw new Error("Missing #canvas-primary element in document.");
    }

    const vpPrimary = new CityWebGL2Viewport(
      canvasPrimary,
      scene,
      staticBuffers,
      idxPrimary,
      sharedVehicleMesh,
      sharedDecalQuads
    );
    const vpCompare = hasCompare && canvasCompare
      ? new CityWebGL2Viewport(
          canvasCompare,
          scene,
          staticBuffers,
          idxCompare,
          sharedVehicleMesh,
          sharedDecalQuads
        )
      : null;

    // Smart Initial Framing around signalized downtown core (~42 deg isometric pitch)
    const focus = computeSmartFocusBounds(scene);
    const focusDiag = Math.hypot(focus.spanX, focus.spanY);
    const minOverviewDist = focusDiag > 360.0 ? 295.0 : 165.0;
    const initDist = Math.max(minOverviewDist, Math.min(415.0, focusDiag * 0.50));
    const initYaw = -Math.PI * 0.64;
    const initPitch = (42.0 * Math.PI) / 180.0;

    const camGoal = {
      x: focus.centerX,
      y: focus.centerY,
      yaw: initYaw,
      pitch: initPitch,
      distance: initDist,
    };
    const camCurr = {
      x: focus.centerX,
      y: focus.centerY,
      yaw: initYaw,
      pitch: initPitch,
      distance: initDist,
    };
    const chaseState = {
      initialized: false,
      x: focus.centerX,
      y: focus.centerY,
      yaw: 0.0,
    };

    const opts = {
      playing: true,
      speedMult: 1.0,
      cameraMode: "orbit", // "orbit" | "ortho" | "chase"
      layoutMode: hasCompare ? "split" : "primary",
      showBuildings: true,
      showDBN: true,
      selectedVehId: null,
    };

    const tStart = idxPrimary.tStart;
    const durationSec = Math.max(
      0.5,
      idxPrimary.duration,
      idxCompare ? idxCompare.duration : 0.0
    );
    const tEnd = tStart + durationSec;
    // On full-length urban replays whose Frame 0 is pre-spawn warmup (0 vehicles), open right at active rush flow
    let simTime =
      idxPrimary.frames.length > 40 && idxPrimary.frames[0].n === 0 && durationSec >= 90.0
        ? tStart + Math.min(90.0, durationSec * 0.18)
        : tStart;

    // Grab DOM Elements (matching @viz-ui #116 / #118 1:1, with legacy fallbacks)
    const stageEl = document.getElementById("viewport-grid") || document.getElementById("viewport-stage");
    const panePrimary = document.getElementById("pane-primary");
    const paneCompare = document.getElementById("pane-compare");
    const scenarioTitleEl = document.getElementById("scenario-title") || document.getElementById("badge-network");
    const layoutGroupEl = document.getElementById("layout-group");
    const btnPlay = document.getElementById("btn-play");
    const timeReadoutEl = document.getElementById("time-readout") || document.getElementById("lbl-clock");
    const timelineSlider = document.getElementById("timeline-slider") || document.getElementById("slider-scrub");
    const incidentWindowEl = document.getElementById("incident-window");
    const btnSpeed = document.getElementById("btn-speed");
    const selectSpeedLegacy = document.getElementById("select-speed");
    const toggleArcsBtn = document.getElementById("toggle-arcs");
    const arcsLegendEl = document.getElementById("arcs-legend");
    const toggleBldgsBtn = document.getElementById("toggle-buildings");
    const btnExportPng = document.getElementById("btn-export-png");
    const btnResetCam = document.getElementById("btn-reset-cam");

    const kpiPLive = document.getElementById("kpi-p-live");
    const kpiCLive = document.getElementById("kpi-c-live");
    const kpiPDelay = document.getElementById("kpi-p-delay");
    const kpiCDelay = document.getElementById("kpi-c-delay");

    const chipEl = document.getElementById("vehicle-chip");
    const chipIdEl = document.getElementById("chip-id");
    const chipModeEl = document.getElementById("chip-mode");
    const chipSpeedEl = document.getElementById("chip-speed");
    const chipDelayEl = document.getElementById("chip-delay");
    const chipCloseBtn = document.getElementById("chip-close");

    if (scenarioTitleEl) {
      scenarioTitleEl.textContent = formatScenarioTitle(primaryTrace.scenario_id, scene.network_id);
    }

    populateStaticScorecards(primaryTrace, compareTrace);

    if (incidentWindowEl && String(primaryTrace.scenario_id || "").toLowerCase().indexOf("incident") >= 0 && durationSec >= 180.0) {
      const incStartPct = Math.max(0, Math.min(90, (90.0 / durationSec) * 100.0));
      const incSpanPct = Math.max(8, Math.min(100 - incStartPct, (240.0 / durationSec) * 100.0));
      incidentWindowEl.style.insetInlineStart = incStartPct.toFixed(1) + "%";
      incidentWindowEl.style.inlineSize = incSpanPct.toFixed(1) + "%";
      incidentWindowEl.hidden = false;
    }

    if (!hasCompare && layoutGroupEl) {
      layoutGroupEl.hidden = true;
      const prevSep = layoutGroupEl.previousElementSibling;
      if (prevSep && prevSep.classList.contains("hud-sep")) {
        prevSep.hidden = true;
      }
    }

    function applyLayout() {
      if (!stageEl) return;
      const mode = hasCompare ? opts.layoutMode : "primary";
      if (stageEl.id === "viewport-grid") {
        stageEl.dataset.layout = mode;
      } else {
        // Legacy #viewport-stage fallback
        stageEl.dataset.layout = mode === "split" ? "split" : "single";
        if (panePrimary) panePrimary.style.display = mode === "compare" ? "none" : "flex";
        if (paneCompare) paneCompare.style.display = mode === "primary" ? "none" : "flex";
      }
      document.querySelectorAll("[data-layout]").forEach((btn) => {
        if (btn === stageEl) return;
        const active = btn.dataset.layout === mode;
        btn.classList.toggle("active", active);
        btn.setAttribute("aria-pressed", active ? "true" : "false");
      });
      vpPrimary._sizeDirty = true;
      if (vpCompare) vpCompare._sizeDirty = true;
    }

    function syncCameraButtons() {
      document.querySelectorAll("[data-cam]").forEach((btn) => {
        const active = btn.dataset.cam === opts.cameraMode;
        btn.classList.toggle("active", active);
        btn.setAttribute("aria-pressed", active ? "true" : "false");
      });
    }

    function syncPlayButton() {
      if (!btnPlay) return;
      btnPlay.dataset.playing = opts.playing ? "true" : "false";
      btnPlay.setAttribute("aria-label", opts.playing ? "Pause playback" : "Play playback");
      if (!btnPlay.querySelector("svg")) {
        btnPlay.textContent = opts.playing ? "Pause" : "Play";
      }
    }

    function syncTimelineUI() {
      if (timeReadoutEl) {
        const relSec = Math.max(0.0, simTime - tStart);
        const txt = formatMMSS(relSec) + " / " + formatMMSS(durationSec);
        if (timeReadoutEl.textContent !== txt) {
          timeReadoutEl.textContent = txt;
        }
      }
      if (timelineSlider && document.activeElement !== timelineSlider) {
        const maxAttr = parseFloat(timelineSlider.max) || 1000.0;
        const norm = durationSec > 0 ? Math.max(0.0, Math.min(1.0, (simTime - tStart) / durationSec)) : 0.0;
        const val = norm * maxAttr;
        timelineSlider.value = val.toFixed(2);
        timelineSlider.style.setProperty("--control-value", val.toFixed(2));
        timelineSlider.style.setProperty("--slider-progress", (norm * 100.0).toFixed(2) + "%");
      }
    }

    // Interactive Synchronized Orbit / Pan / Cursor-Centered Wheel Zoom / Touch Pinch-Pan
    const eyeScratch = new Float32Array(3);
    const targetScratch = new Float32Array(3);

    function computeCurrentEyeAndTarget(dtSec) {
      const alpha = dtSec > 0 ? 1.0 - Math.exp(-16.0 * dtSec) : 1.0;
      camCurr.x += (camGoal.x - camCurr.x) * alpha;
      camCurr.y += (camGoal.y - camCurr.y) * alpha;
      camCurr.yaw += wrapAngle(camGoal.yaw - camCurr.yaw) * alpha;
      camCurr.pitch += (camGoal.pitch - camCurr.pitch) * alpha;
      camCurr.distance += (camGoal.distance - camCurr.distance) * alpha;

      if (opts.cameraMode === "chase") {
        let snap = opts.selectedVehId ? vpPrimary.getVehicleSnapshotById(opts.selectedVehId) : null;
        if (!snap && vpCompare && opts.selectedVehId) {
          snap = vpCompare.getVehicleSnapshotById(opts.selectedVehId);
        }
        if (!snap) {
          const nextLead = vpPrimary.findBestLeadVehicleId(focus.centerX, focus.centerY, vpCompare);
          if (nextLead) {
            opts.selectedVehId = nextLead;
            snap = vpPrimary.getVehicleSnapshotById(nextLead);
          }
        }
        if (snap) {
          const followAlpha = !chaseState.initialized || dtSec <= 0 ? 1.0 : 1.0 - Math.exp(-6.5 * dtSec);
          chaseState.x += (snap.x - chaseState.x) * followAlpha;
          chaseState.y += (snap.y - chaseState.y) * followAlpha;
          chaseState.yaw += wrapAngle(snap.yaw - chaseState.yaw) * (chaseState.initialized ? 1.0 - Math.exp(-5.0 * dtSec) : 1.0);
          chaseState.initialized = true;

          const c = Math.cos(chaseState.yaw), s = Math.sin(chaseState.yaw);
          targetScratch[0] = chaseState.x + c * 9.0;
          targetScratch[1] = chaseState.y + s * 9.0;
          targetScratch[2] = 1.4;
          eyeScratch[0] = chaseState.x - c * 28.0;
          eyeScratch[1] = chaseState.y - s * 28.0;
          eyeScratch[2] = 14.5;
          return { isOrtho: false, enableFade: true, dist: 32.0 };
        }
      } else {
        chaseState.initialized = false;
      }

      if (opts.cameraMode === "ortho") {
        targetScratch[0] = camCurr.x;
        targetScratch[1] = camCurr.y;
        targetScratch[2] = 0.0;
        eyeScratch[0] = camCurr.x;
        eyeScratch[1] = camCurr.y;
        eyeScratch[2] = Math.max(120.0, camCurr.distance * 1.25);
        return { isOrtho: true, enableFade: false, dist: camCurr.distance };
      }

      const horiz = camCurr.distance * Math.cos(camCurr.pitch);
      const ez = camCurr.distance * Math.sin(camCurr.pitch);
      targetScratch[0] = camCurr.x;
      targetScratch[1] = camCurr.y;
      targetScratch[2] = 0.0;
      eyeScratch[0] = camCurr.x + horiz * Math.cos(camCurr.yaw);
      eyeScratch[1] = camCurr.y + horiz * Math.sin(camCurr.yaw);
      eyeScratch[2] = Math.max(8.0, ez);
      return {
        isOrtho: false,
        enableFade: camCurr.pitch < 0.88 && camCurr.distance < 165.0,
        dist: camCurr.distance,
      };
    }

    function panWorldByScreenDelta(dxPx, dyPx) {
      const panScale = camGoal.distance * 0.00145;
      if (opts.cameraMode === "ortho") {
        camGoal.x -= dxPx * panScale;
        camGoal.y += dyPx * panScale;
        return;
      }
      // Eye = target + horiz*(cos(yaw), sin(yaw)) => forward f = (-cos(yaw), -sin(yaw)), right s = (-sin(yaw), cos(yaw))
      const rx = -Math.sin(camCurr.yaw), ry = Math.cos(camCurr.yaw);
      const fx = -Math.cos(camCurr.yaw), fy = -Math.sin(camCurr.yaw);
      const pitchComp = 1.0 / Math.max(0.38, Math.sin(camCurr.pitch));
      camGoal.x += (-rx * dxPx + fx * dyPx * pitchComp) * panScale;
      camGoal.y += (-ry * dxPx + fy * dyPx * pitchComp) * panScale;
    }

    function bindCanvasPointerControls(canvas, viewport) {
      let dragging = false;
      let dragButton = 0;
      let lastX = 0, lastY = 0;
      let movedPx = 0;

      canvas.addEventListener("contextmenu", (ev) => ev.preventDefault());

      canvas.addEventListener("mousedown", (ev) => {
        dragging = true;
        dragButton = ev.button;
        lastX = ev.clientX;
        lastY = ev.clientY;
        movedPx = 0;
      });

      window.addEventListener("mousemove", (ev) => {
        if (!dragging) return;
        const dx = ev.clientX - lastX;
        const dy = ev.clientY - lastY;
        lastX = ev.clientX;
        lastY = ev.clientY;
        movedPx += Math.hypot(dx, dy);

        if (opts.cameraMode === "chase" && movedPx > 6.0) {
          opts.cameraMode = "orbit";
          syncCameraButtons();
        }

        if (dragButton === 2 || dragButton === 1 || ev.shiftKey || ev.metaKey || opts.cameraMode === "ortho") {
          panWorldByScreenDelta(dx, dy);
        } else {
          camGoal.yaw -= dx * 0.0062;
          camGoal.pitch = Math.max(0.20, Math.min(1.46, camGoal.pitch + dy * 0.0052));
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
          updateVehicleChip();
        }
      });

      canvas.addEventListener("dblclick", (ev) => {
        const picked = viewport.pickVehicleAtClientXY(ev.clientX, ev.clientY);
        if (picked) {
          opts.selectedVehId = picked;
          opts.cameraMode = "chase";
          syncCameraButtons();
          updateVehicleChip();
        }
      });

      // Touch Orbit + Two-Finger Pan & Pinch Zoom
      let prevTouches = null;
      canvas.addEventListener("touchstart", (ev) => {
        prevTouches = Array.from(ev.touches).map((t) => ({ x: t.clientX, y: t.clientY }));
      }, { passive: true });

      canvas.addEventListener("touchmove", (ev) => {
        if (!prevTouches || ev.touches.length === 0) return;
        ev.preventDefault();
        if (ev.touches.length === 1 && prevTouches.length === 1) {
          const dx = ev.touches[0].clientX - prevTouches[0].x;
          const dy = ev.touches[0].clientY - prevTouches[0].y;
          if (opts.cameraMode === "ortho") {
            panWorldByScreenDelta(dx, dy);
          } else {
            camGoal.yaw -= dx * 0.0062;
            camGoal.pitch = Math.max(0.20, Math.min(1.46, camGoal.pitch + dy * 0.0052));
          }
        } else if (ev.touches.length >= 2 && prevTouches.length >= 2) {
          const cx0 = 0.5 * (prevTouches[0].x + prevTouches[1].x);
          const cy0 = 0.5 * (prevTouches[0].y + prevTouches[1].y);
          const cx1 = 0.5 * (ev.touches[0].clientX + ev.touches[1].clientX);
          const cy1 = 0.5 * (ev.touches[0].clientY + ev.touches[1].clientY);
          panWorldByScreenDelta(cx1 - cx0, cy1 - cy0);

          const span0 = Math.hypot(prevTouches[1].x - prevTouches[0].x, prevTouches[1].y - prevTouches[0].y);
          const span1 = Math.hypot(ev.touches[1].clientX - ev.touches[0].clientX, ev.touches[1].clientY - ev.touches[0].clientY);
          if (span0 > 8.0 && span1 > 8.0) {
            camGoal.distance = Math.max(24.0, Math.min(2800.0, camGoal.distance * (span0 / span1)));
          }
        }
        prevTouches = Array.from(ev.touches).map((t) => ({ x: t.clientX, y: t.clientY }));
      }, { passive: false });

      // Cursor-Centered Wheel Zoom
      canvas.addEventListener("wheel", (ev) => {
        ev.preventDefault();
        if (opts.cameraMode === "chase") {
          opts.cameraMode = "orbit";
          syncCameraButtons();
        }
        const rect = canvas.getBoundingClientRect();
        const ndcX = rect.width > 1 ? ((ev.clientX - rect.left) / rect.width) * 2.0 - 1.0 : 0.0;
        const ndcY = rect.height > 1 ? 1.0 - ((ev.clientY - rect.top) / rect.height) * 2.0 : 0.0;
        const aspect = rect.width / Math.max(1.0, rect.height);

        const factor = Math.exp(Math.max(-0.35, Math.min(0.35, ev.deltaY * 0.0012)));
        const nextDist = Math.max(24.0, Math.min(2800.0, camGoal.distance * factor));
        const appliedRatio = nextDist / camGoal.distance;
        camGoal.distance = nextDist;

        if (appliedRatio < 0.999) {
          const hit = intersectCursorGroundZ0(
            ndcX,
            ndcY,
            eyeScratch,
            targetScratch,
            (44.0 * Math.PI) / 180.0,
            aspect,
            opts.cameraMode === "ortho",
            camCurr.distance * 0.52
          );
          if (hit) {
            const pull = (1.0 - appliedRatio) * 0.65;
            const shiftX = Math.max(-180.0, Math.min(180.0, (hit[0] - camGoal.x) * pull));
            const shiftY = Math.max(-180.0, Math.min(180.0, (hit[1] - camGoal.y) * pull));
            camGoal.x += shiftX;
            camGoal.y += shiftY;
          }
        }
      }, { passive: false });
    }

    bindCanvasPointerControls(canvasPrimary, vpPrimary);
    if (vpCompare && canvasCompare) {
      bindCanvasPointerControls(canvasCompare, vpCompare);
    }

    // Wire Minimalist Floating HUD Controls
    if (btnPlay) {
      btnPlay.addEventListener("click", () => {
        opts.playing = !opts.playing;
        syncPlayButton();
      });
    }

    window.addEventListener("keydown", (ev) => {
      if (ev.code === "Space" && !ev.metaKey && !ev.ctrlKey && !ev.altKey) {
        const tag = ev.target && ev.target.tagName ? ev.target.tagName.toUpperCase() : "";
        if (tag === "BUTTON" || tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") return;
        ev.preventDefault();
        opts.playing = !opts.playing;
        syncPlayButton();
      } else if (ev.key === "Escape" && opts.selectedVehId) {
        opts.selectedVehId = null;
        if (opts.cameraMode === "chase") {
          opts.cameraMode = "orbit";
          syncCameraButtons();
        }
        updateVehicleChip();
      }
    });

    if (timelineSlider) {
      timelineSlider.addEventListener("input", () => {
        const maxAttr = parseFloat(timelineSlider.max) || 1000.0;
        const rawVal = parseFloat(timelineSlider.value) || 0.0;
        const norm = Math.max(0.0, Math.min(1.0, rawVal / maxAttr));
        simTime = tStart + norm * durationSec;
        renderSingleTick(0.0, performance.now() * 0.001);
        updateLiveHUD();
      });
    }

    const SPEED_STEPS = [1, 2, 4];
    if (btnSpeed) {
      btnSpeed.addEventListener("click", () => {
        const idx = SPEED_STEPS.indexOf(opts.speedMult);
        const next = SPEED_STEPS[(idx + 1) % SPEED_STEPS.length];
        opts.speedMult = next;
        btnSpeed.dataset.speed = String(next);
        btnSpeed.textContent = next + "×";
        btnSpeed.setAttribute("aria-label", "Playback speed " + next + "x");
      });
    }
    if (selectSpeedLegacy) {
      selectSpeedLegacy.addEventListener("change", () => {
        opts.speedMult = parseFloat(selectSpeedLegacy.value) || 1.0;
      });
    }

    document.querySelectorAll("[data-layout]").forEach((btn) => {
      if (btn === stageEl) return;
      btn.addEventListener("click", () => {
        const req = btn.dataset.layout;
        if (!req) return;
        opts.layoutMode = req;
        applyLayout();
      });
    });

    document.querySelectorAll("[data-cam]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const req = btn.dataset.cam;
        if (!req) return;
        if (req === "chase" && !opts.selectedVehId) {
          opts.selectedVehId = vpPrimary.findBestLeadVehicleId(focus.centerX, focus.centerY, vpCompare);
        } else if (req === "orbit" && opts.cameraMode === "orbit") {
          // Clicking active "3D View" smoothly re-centers on the smart downtown focus
          camGoal.x = focus.centerX;
          camGoal.y = focus.centerY;
          camGoal.yaw = initYaw;
          camGoal.pitch = initPitch;
          camGoal.distance = initDist;
        }
        opts.cameraMode = req;
        syncCameraButtons();
        updateVehicleChip();
      });
    });

    if (btnResetCam) {
      btnResetCam.addEventListener("click", () => {
        camGoal.x = focus.centerX;
        camGoal.y = focus.centerY;
        camGoal.yaw = initYaw;
        camGoal.pitch = initPitch;
        camGoal.distance = initDist;
        opts.cameraMode = "orbit";
        syncCameraButtons();
      });
    }

    if (toggleArcsBtn) {
      toggleArcsBtn.addEventListener("click", () => {
        opts.showDBN = !opts.showDBN;
        toggleArcsBtn.classList.toggle("active", opts.showDBN);
        toggleArcsBtn.setAttribute("aria-pressed", opts.showDBN ? "true" : "false");
        if (arcsLegendEl) {
          arcsLegendEl.style.opacity = opts.showDBN ? "" : "0.38";
        }
        if (!opts.playing) renderSingleTick(0.0, performance.now() * 0.001);
      });
    }

    if (toggleBldgsBtn) {
      toggleBldgsBtn.addEventListener("click", () => {
        opts.showBuildings = !opts.showBuildings;
        toggleBldgsBtn.classList.toggle("active", opts.showBuildings);
        toggleBldgsBtn.setAttribute("aria-pressed", opts.showBuildings ? "true" : "false");
        if (!opts.playing) renderSingleTick(0.0, performance.now() * 0.001);
      });
    }

    if (chipCloseBtn) {
      chipCloseBtn.addEventListener("click", () => {
        opts.selectedVehId = null;
        if (opts.cameraMode === "chase") {
          opts.cameraMode = "orbit";
          syncCameraButtons();
        }
        updateVehicleChip();
      });
    }

    if (btnExportPng) {
      btnExportPng.addEventListener("click", () => {
        const showBoth = hasCompare && opts.layoutMode === "split" && vpCompare;
        const srcLeft = opts.layoutMode === "compare" && vpCompare ? canvasCompare : canvasPrimary;
        const w1 = srcLeft.width;
        const h1 = srcLeft.height;
        const w2 = showBoth ? canvasCompare.width : 0;
        const outCanvas = document.createElement("canvas");
        outCanvas.width = w1 + (showBoth ? w2 + 2 : 0);
        outCanvas.height = h1 + 52;
        const ctx = outCanvas.getContext("2d");
        ctx.fillStyle = "#080c14";
        ctx.fillRect(0, 0, outCanvas.width, outCanvas.height);
        ctx.drawImage(srcLeft, 0, 52);
        if (showBoth) {
          ctx.fillStyle = "rgba(255,255,255,0.14)";
          ctx.fillRect(w1, 52, 2, h1);
          ctx.drawImage(canvasCompare, w1 + 2, 52);
        }
        ctx.fillStyle = "#f8fafc";
        ctx.font = "600 20px -apple-system, BlinkMacSystemFont, system-ui, sans-serif";
        const label =
          formatScenarioTitle(primaryTrace.scenario_id, scene.network_id) +
          "  ·  " +
          formatMMSS(simTime - tStart);
        ctx.fillText(label, 20, 33);
        const a = document.createElement("a");
        a.download = "swarm_3d_" + (scene.network_id || "city") + "_t" + Math.round(simTime) + ".png";
        a.href = outCanvas.toDataURL("image/png");
        a.click();
      });
    }

    let prevLiveP = "";
    let prevLiveC = "";

    function updateVehicleChip() {
      if (!chipEl) return;
      if (!opts.selectedVehId) {
        chipEl.hidden = true;
        return;
      }
      const snap =
        vpPrimary.getVehicleSnapshotById(opts.selectedVehId) ||
        (vpCompare ? vpCompare.getVehicleSnapshotById(opts.selectedVehId) : null);
      if (!snap) {
        chipEl.hidden = true;
        return;
      }
      chipEl.hidden = false;
      if (chipIdEl) chipIdEl.textContent = snap.id;
      if (chipModeEl) {
        const meta = MODE_META[String(snap.mode)] || MODE_META["1"];
        chipModeEl.textContent = meta.label;
        chipModeEl.dataset.mode = meta.slug;
      }
      if (chipSpeedEl) chipSpeedEl.textContent = snap.v.toFixed(1) + " m/s";
      if (chipDelayEl) chipDelayEl.textContent = Math.round(snap.d_post * 100.0) + "% risk";
    }

    function updateLiveHUD() {
      syncTimelineUI();
      const fP = vpPrimary.currentFrameObj;
      if (fP && fP.kpi) {
        const k = fP.kpi;
        const liveStr = k.active_veh + " active · " + k.mean_speed_mps.toFixed(1) + " m/s";
        if (kpiPLive && liveStr !== prevLiveP) {
          kpiPLive.textContent = liveStr;
          prevLiveP = liveStr;
        }
        if (!primaryTrace.summary_metrics && kpiPDelay) {
          kpiPDelay.textContent = k.mean_speed_mps.toFixed(1) + " m/s";
        }
      }
      if (vpCompare && vpCompare.currentFrameObj && vpCompare.currentFrameObj.kpi) {
        const kC = vpCompare.currentFrameObj.kpi;
        const liveCStr = kC.active_veh + " active · " + kC.mean_speed_mps.toFixed(1) + " m/s";
        if (kpiCLive && liveCStr !== prevLiveC) {
          kpiCLive.textContent = liveCStr;
          prevLiveC = liveCStr;
        }
        if (compareTrace && !compareTrace.summary_metrics && kpiCDelay) {
          kpiCDelay.textContent = kC.mean_speed_mps.toFixed(1) + " m/s";
        }
      }
      if (opts.selectedVehId) {
        updateVehicleChip();
      }
    }

    function renderSingleTick(dtSec, wallSec) {
      const camInfo = computeCurrentEyeAndTarget(dtSec);
      const selectedSlotP = opts.selectedVehId
        ? (idxPrimary.idToSlot.get(opts.selectedVehId) ?? -1)
        : -1;

      const showP = !hasCompare || opts.layoutMode === "split" || opts.layoutMode === "primary";
      const showC = hasCompare && vpCompare && (opts.layoutMode === "split" || opts.layoutMode === "compare");

      if (showP) {
        vpPrimary.render(
          eyeScratch,
          targetScratch,
          camInfo.dist,
          camInfo.isOrtho,
          camInfo.enableFade,
          simTime,
          wallSec,
          opts,
          selectedSlotP
        );
      }
      if (showC) {
        const selectedSlotC = opts.selectedVehId
          ? (idxCompare.idToSlot.get(opts.selectedVehId) ?? -1)
          : -1;
        vpCompare.render(
          eyeScratch,
          targetScratch,
          camInfo.dist,
          camInfo.isOrtho,
          camInfo.enableFade,
          simTime,
          wallSec,
          opts,
          selectedSlotC
        );
      }
    }

    // Apply layout, render Frame 0 immediately, and fade out #loading-overlay!
    applyLayout();
    syncCameraButtons();
    syncPlayButton();
    renderSingleTick(0.0, performance.now() * 0.001);
    updateLiveHUD();

    const loadTimeMs = Math.round((performance.now() - bootStartMs) * 100) / 100;
    const loadingOverlay = document.getElementById("loading-overlay");
    if (loadingOverlay) {
      loadingOverlay.classList.add("is-hidden");
      loadingOverlay.setAttribute("aria-hidden", "true");
    }

    // Rolling FPS meter & 60-120 Hz continuous animation loop
    let lastWallMs = performance.now();
    let fpsSmooth = 60.0;
    let lastHudUpdateMs = 0.0;

    function animationLoop(nowMs) {
      const rawDt = (nowMs - lastWallMs) * 0.001;
      lastWallMs = nowMs;
      if (rawDt > 0.001 && rawDt < 0.25) {
        const instFps = 1.0 / rawDt;
        fpsSmooth = fpsSmooth * 0.90 + instFps * 0.10;
      }
      const elapsedS = Math.min(0.08, Math.max(0.0, rawDt));

      if (opts.playing && durationSec > 0) {
        simTime += elapsedS * opts.speedMult;
        if (simTime > tEnd) {
          simTime = tStart + ((simTime - tStart) % durationSec);
        }
      }

      renderSingleTick(elapsedS, nowMs * 0.001);

      if (nowMs - lastHudUpdateMs >= 50.0) {
        lastHudUpdateMs = nowMs;
        updateLiveHUD();
      }

      requestAnimationFrame(animationLoop);
    }

    requestAnimationFrame(animationLoop);

    // Expose rich diagnostic & automation API for real-browser QA
    const controllerApi = {
      loadTimeMs,
      getFps: () => Math.round(fpsSmooth * 10) / 10,
      getSimTime: () => simTime,
      setSimTime: (targetSec) => {
        simTime = Math.max(tStart, Math.min(tEnd, Number(targetSec) || 0));
        renderSingleTick(0.0, performance.now() * 0.001);
        updateLiveHUD();
        return simTime;
      },
      setPlaying: (flag) => {
        opts.playing = Boolean(flag);
        syncPlayButton();
      },
      getSampledVehicles: (which) => {
        if (which === "compare" && vpCompare) return vpCompare.snapshotAllActiveVehicles();
        return vpPrimary.snapshotAllActiveVehicles();
      },
      getStats: () => ({
        loadTimeMs,
        fps: Math.round(fpsSmooth * 10) / 10,
        simTime,
        tStart,
        tEnd,
        durationSec,
        layoutMode: opts.layoutMode,
        cameraMode: opts.cameraMode,
        primaryVehicles: vpPrimary.activeInstanceCount,
        compareVehicles: vpCompare ? vpCompare.activeInstanceCount : 0,
        glErrorPrimary: vpPrimary.gl.getError(),
        glErrorCompare: vpCompare ? vpCompare.gl.getError() : 0,
      }),
      getUnpackedPayload: () => ({
        scene,
        primary_trace: unpackSoATrace(primaryTrace),
        comparison_trace: hasCompare ? unpackSoATrace(compareTrace) : null,
        meta: payload.meta,
      }),
      vpPrimary,
      vpCompare,
      opts,
      focus,
      camGoal,
      camCurr,
    };

    if (typeof window !== "undefined") {
      window.__SWARM_VIEWER__ = controllerApi;
    }
    return controllerApi;
  }

  // ===========================================================================
  // 11. Bulletproof Async Boot Entrypoint & Fail-Loud Error Banner
  // ===========================================================================

  let _bootPromise = null;

  function showFatalErrorBanner(err) {
    const msg = err && err.message ? err.message : String(err);
    const loadingOverlay = document.getElementById("loading-overlay");
    if (loadingOverlay) {
      loadingOverlay.classList.add("is-hidden");
      loadingOverlay.setAttribute("aria-hidden", "true");
    }
    const banner = document.getElementById("error-banner");
    const msgEl = document.getElementById("error-message");
    if (banner) {
      banner.hidden = false;
      banner.style.display = "block";
    }
    if (msgEl) {
      msgEl.textContent = msg;
    }
  }

  async function bootCityDBNViewer(explicitPayload) {
    if (_bootPromise && !explicitPayload) {
      return _bootPromise;
    }
    const run = async () => {
      const t0 = performance.now();
      const statusEl = document.getElementById("loading-status");
      const setStatus = (txt) => {
        if (statusEl) statusEl.textContent = txt;
      };
      try {
        const normalized = explicitPayload
          ? normalizePayload(explicitPayload, null)
          : await decodeSwarmPayloadFromDOM(setStatus);
        setStatus("Compiling WebGL2 shaders & 3D city meshes…");
        await yieldToMain();
        return mountCityDBNViewerInstance(normalized, t0);
      } catch (err) {
        showFatalErrorBanner(err);
        throw err;
      }
    };
    _bootPromise = run();
    return _bootPromise;
  }

  function mountCityDBNViewer(payload) {
    return bootCityDBNViewer(payload);
  }

  if (typeof window !== "undefined") {
    window.bootCityDBNViewer = bootCityDBNViewer;
    window.mountCityDBNViewer = mountCityDBNViewer;
    window.decodeSwarmPayload = decodeSwarmPayload;
    window.unpackSoATrace = unpackSoATrace;

    const autoStart = () => {
      const hasGz = document.getElementById("swarm-payload-gz");
      const hasJson = document.getElementById("swarm-payload");
      if (hasGz || hasJson) {
        bootCityDBNViewer().catch(() => {
          // Error already surfaced visibly via #error-banner
        });
      }
    };

    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", autoStart, { once: true });
    } else {
      queueMicrotask(autoStart);
    }
  }
})();
