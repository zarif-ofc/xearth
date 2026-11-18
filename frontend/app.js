/**
 * PyTracker — Minimal 3D Earth with Thumb & Index Fingertip Tracking
 * 100% Client-Side (MediaPipe Tasks Vision WASM/GPU + Three.js PBR Earth)
 * Pure Black Background • Soft Satellite Reflection • Zero Backend Dependency
 */

// ---------------------------------------------------------------------------
// 1. Configuration & Constants
// ---------------------------------------------------------------------------
const MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
const WASM_CDN_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm";
const THREE_CDN = "https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.js";
const EARTH_TEX_BASE = "https://raw.githubusercontent.com/mrdoob/three.js/dev/examples/textures/planets/";

// Only Thumb (4) and Index (8) fingertips show dots
const FINGERTIP_INDICES = [4, 8];

// ---------------------------------------------------------------------------
// 2. 1€ Filter (One Euro Filter) for Adaptive Anti-Jitter
// ---------------------------------------------------------------------------
class LowPassFilter {
  constructor(alpha = 1.0, initVal = 0.0) {
    this.s = initVal;
    this.hasInit = false;
    this.setAlpha(alpha);
  }

  setAlpha(alpha) {
    this.alpha = Math.max(0.0, Math.min(1.0, alpha));
  }

  filter(val) {
    if (!this.hasInit) {
      this.s = val;
      this.hasInit = true;
      return val;
    }
    this.s = this.alpha * val + (1.0 - this.alpha) * this.s;
    return this.s;
  }

  reset() {
    this.hasInit = false;
    this.s = 0.0;
  }
}

class OneEuroFilter {
  constructor({ minCutoff = 1.2, beta = 0.015, dCutoff = 1.0 } = {}) {
    this.minCutoff = minCutoff;
    this.beta = beta;
    this.dCutoff = dCutoff;
    this.xFilter = new LowPassFilter();
    this.dxFilter = new LowPassFilter();
    this.lastTime = null;
    this.lastValue = null;
  }

  alpha(rate, cutoff) {
    const tau = 1.0 / (2.0 * Math.PI * cutoff);
    const te = 1.0 / rate;
    return 1.0 / (1.0 + tau / te);
  }

  filter(val, timestamp) {
    if (this.lastTime === null || timestamp <= this.lastTime) {
      this.lastTime = timestamp;
      this.lastValue = val;
      this.xFilter.reset();
      this.dxFilter.reset();
      this.xFilter.filter(val);
      return val;
    }

    const dt = Math.max(1e-5, timestamp - this.lastTime);
    const rate = 1.0 / dt;
    this.lastTime = timestamp;

    const dx = (val - this.lastValue) / dt;
    this.lastValue = val;

    this.dxFilter.setAlpha(this.alpha(rate, this.dCutoff));
    const edx = this.dxFilter.filter(dx);

    const cutoff = this.minCutoff + this.beta * Math.abs(edx);
    this.xFilter.setAlpha(this.alpha(rate, cutoff));
    return this.xFilter.filter(val);
  }

  reset() {
    this.lastTime = null;
    this.lastValue = null;
    this.xFilter.reset();
    this.dxFilter.reset();
  }
}

function createHandSlot() {
  return {
    active: false,
    opacity: 0.0,
    pinchScale: 0.0,
    lastDetectedTime: 0,
    targetLandmarks: null,
    renderedLandmarks: null,
    filters: Array.from({ length: 21 }, () => ({
      x: new OneEuroFilter({ minCutoff: 1.2, beta: 0.015, dCutoff: 1.0 }),
      y: new OneEuroFilter({ minCutoff: 1.2, beta: 0.015, dCutoff: 1.0 }),
      z: new OneEuroFilter({ minCutoff: 1.2, beta: 0.015, dCutoff: 1.0 }),
    })),
    resetFilters() {
      for (let i = 0; i < 21; i++) {
        this.filters[i].x.reset();
        this.filters[i].y.reset();
        this.filters[i].z.reset();
      }
    },
  };
}

const handSlots = [createHandSlot(), createHandSlot()];

// ---------------------------------------------------------------------------
// 3. Three.js 3D Earth Globe State
// ---------------------------------------------------------------------------
const earth = {
  THREE: null,
  renderer: null,
  scene: null,
  camera: null,
  earthGroup: null,
  earthMesh: null,
  cloudMesh: null,
  atmMesh: null,
  rotVelX: 0,
  rotVelY: 0,
  targetCamZ: 3.8,
  currentCamZ: 3.8,
  ready: false,
};

const pinchState = {
  hands: [
    { isPinching: false, midX: 0, midY: 0, prevMidX: 0, prevMidY: 0, wasPinching: false },
    { isPinching: false, midX: 0, midY: 0, prevMidX: 0, prevMidY: 0, wasPinching: false },
  ],
  prevTwoHandDist: null,
};

// ---------------------------------------------------------------------------
// 4. Application State
// ---------------------------------------------------------------------------
const state = {
  handLandmarker: null,
  webcamRunning: false,
  stream: null,
  lastVideoTime: -1,
  animationFrameId: null,
  lastFrameTimestamp: performance.now(),
  mirrorMode: true,
};

// ---------------------------------------------------------------------------
// 5. DOM Elements
// ---------------------------------------------------------------------------
const video = document.getElementById("webcam");
const canvas = document.getElementById("output_canvas");
const ctx = canvas.getContext("2d", { willReadFrequently: false, alpha: true });
const earthCanvas = document.getElementById("earth-canvas");
const loadingOverlay = document.getElementById("loading-overlay");

// ---------------------------------------------------------------------------
// 6. Viewport Sizing (Full Screen)
// ---------------------------------------------------------------------------
function resize() {
  const w = window.innerWidth;
  const h = window.innerHeight;

  canvas.width = w;
  canvas.height = h;

  if (earth.renderer && earth.camera) {
    earth.renderer.setSize(w, h, false);
    earth.camera.aspect = w / h;
    earth.camera.updateProjectionMatrix();
  }
}

// ---------------------------------------------------------------------------
// 7. Three.js 3D Earth Setup (Soft Satellite Reflection & Subtle Micro-Detail)
// ---------------------------------------------------------------------------
async function initEarthGlobe() {
  try {
    const THREE = await import(THREE_CDN);
    earth.THREE = THREE;

    // WebGL Renderer with transparent clear color revealing black background
    const renderer = new THREE.WebGLRenderer({
      canvas: earthCanvas,
      alpha: true,
      antialias: true,
    });
    renderer.setClearColor(0x000000, 0);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    earth.renderer = renderer;

    // Scene
    const scene = new THREE.Scene();
    earth.scene = scene;

    // Camera
    const w = window.innerWidth;
    const h = window.innerHeight;
    const camera = new THREE.PerspectiveCamera(50, w / h, 0.1, 100);
    camera.position.z = earth.currentCamZ;
    earth.camera = camera;
    renderer.setSize(w, h, false);

    // Sunlight: Graze curvature at (5, 2, 3), intensity lowered to 1.35
    const sunLight = new THREE.DirectionalLight(0xffffff, 1.35);
    sunLight.position.set(5, 2, 3);
    scene.add(sunLight);

    // Ambient light: Soft space fill so dark side is visible
    const ambLight = new THREE.AmbientLight(0x112233, 0.35);
    scene.add(ambLight);

    // Earth group for rotation
    const earthGroup = new THREE.Group();
    scene.add(earthGroup);
    earth.earthGroup = earthGroup;

    // Textures loader
    const loader = new THREE.TextureLoader();
    loader.crossOrigin = "anonymous";

    let earthMat;
    try {
      const [mapTex, specTex, normTex] = await Promise.all([
        loader.loadAsync(EARTH_TEX_BASE + "earth_atmos_2048.jpg"),
        loader.loadAsync(EARTH_TEX_BASE + "earth_specular_2048.jpg"),
        loader.loadAsync(EARTH_TEX_BASE + "earth_normal_2048.jpg"),
      ]);

      // MeshStandardMaterial: Physically-based natural satellite reflection
      // Subtle normalScale (0.35, 0.35) avoids molded plastic appearance
      earthMat = new THREE.MeshStandardMaterial({
        map: mapTex,
        normalMap: normTex,
        normalScale: new THREE.Vector2(0.35, 0.35),
        roughness: 0.78,
        metalness: 0.05,
      });

      // Shader hook: Rich vibrant color grading + soft diffuse ocean sheen
      earthMat.onBeforeCompile = (shader) => {
        // 1. Color Grading inside map_fragment
        shader.fragmentShader = shader.fragmentShader.replace(
          "#include <map_fragment>",
          `
          #include <map_fragment>
          #ifdef USE_MAP
            {
              vec3 col = diffuseColor.rgb;
              float r = col.r;
              float g = col.g;
              float b = col.b;

              // Polar ice & snow
              float minVal = min(r, min(g, b));
              float isIce = clamp((minVal - 0.65) / 0.20, 0.0, 1.0);

              // Water detection
              float waterScore = clamp((b - r * 1.1) / (b + 0.05), 0.0, 1.0);
              if (r < 0.14 && b > 0.10) {
                waterScore = max(waterScore, 0.95);
              }
              float isWater = waterScore * (1.0 - isIce);
              float isLand = (1.0 - isWater) * (1.0 - isIce);

              // Vibrant deep-blue ocean (rich sapphire to radiant cyan coastlines)
              vec3 deepOcean = vec3(0.01, 0.22, 0.80);
              vec3 shallowOcean = vec3(0.00, 0.58, 0.94);
              float depth = clamp((b - 0.15) / 0.30, 0.0, 1.0);
              vec3 vibrantWater = mix(deepOcean, shallowOcean, depth);

              // Vibrant emerald green land and warm golden desert
              float isDesert = clamp((r - 0.35) / 0.20, 0.0, 1.0) * clamp((r - b - 0.10) / 0.20, 0.0, 1.0);
              vec3 lushGreen = vec3(0.04, 0.72, 0.22);
              vec3 deepForest = vec3(0.02, 0.52, 0.14);
              vec3 desertColor = vec3(0.82, 0.60, 0.32);

              float vegIntensity = clamp((g - 0.15) / 0.25, 0.0, 1.0);
              vec3 vegColor = mix(deepForest, lushGreen, vegIntensity);
              vec3 vibrantLand = mix(vegColor, desertColor, isDesert);

              col = mix(col, vibrantWater, isWater * 0.88);
              col = mix(col, vibrantLand, isLand * 0.85);

              diffuseColor.rgb = col;
            }
          #endif
          `
        );

        // 2. Soft, natural roughness in roughnessmap_fragment (diffuse water sheen, matte land)
        shader.fragmentShader = shader.fragmentShader.replace(
          "#include <roughnessmap_fragment>",
          `
          #include <roughnessmap_fragment>
          #ifdef USE_MAP
            {
              float r = diffuseColor.r;
              float g = diffuseColor.g;
              float b = diffuseColor.b;
              float waterScore = clamp((b - r * 1.1) / (b + 0.05), 0.0, 1.0);
              if (r < 0.14 && b > 0.10) waterScore = max(waterScore, 0.95);

              // Soft natural ocean diffusion (0.68) vs matte land (0.95)
              // Eliminates harsh, plasticky cyan specular hotspots
              roughnessFactor = mix(0.95, 0.68, waterScore);
            }
          #endif
          `
        );
      };
    } catch (texErr) {
      console.warn("Using fallback wireframe earth material:", texErr);
      earthMat = new THREE.MeshBasicMaterial({
        color: 0x0ea5e9,
        wireframe: true,
        transparent: true,
        opacity: 0.7,
      });
    }

    const earthGeo = new THREE.SphereGeometry(1.4, 64, 64);
    const earthMesh = new THREE.Mesh(earthGeo, earthMat);
    earthGroup.add(earthMesh);
    earth.earthMesh = earthMesh;

    // Cloud layer: Subtle, soft wisps (opacity 0.22)
    let cloudMesh = null;
    try {
      const cloudTex = await loader.loadAsync(EARTH_TEX_BASE + "earth_clouds_1024.png");
      const cloudMat = new THREE.MeshLambertMaterial({
        map: cloudTex,
        transparent: true,
        opacity: 0.22,
        depthWrite: false,
      });
      const cloudGeo = new THREE.SphereGeometry(1.425, 32, 32);
      cloudMesh = new THREE.Mesh(cloudGeo, cloudMat);
      earthGroup.add(cloudMesh);
    } catch (e) {
      console.warn("Cloud layer skipped:", e);
    }
    earth.cloudMesh = cloudMesh;

    // Soft celestial atmospheric rim glow
    const atmGeo = new THREE.SphereGeometry(1.48, 32, 32);
    const atmMat = new THREE.MeshLambertMaterial({
      color: 0x38bdf8,
      side: THREE.BackSide,
      transparent: true,
      opacity: 0.16,
    });
    const atmMesh = new THREE.Mesh(atmGeo, atmMat);
    earthGroup.add(atmMesh);
    earth.atmMesh = atmMesh;

    earth.ready = true;
    resize();
  } catch (err) {
    console.warn("Failed to initialize Earth globe:", err);
  }
}

// ---------------------------------------------------------------------------
// 8. Pinch Gestures & Earth Physics
// ---------------------------------------------------------------------------
function updatePinchGestures() {
  for (let h = 0; h < 2; h++) {
    const ps = pinchState.hands[h];
    ps.wasPinching = ps.isPinching;
    const slot = handSlots[h];

    if (slot.targetLandmarks && slot.opacity > 0.05) {
      const thumb = slot.targetLandmarks[4];
      const index = slot.targetLandmarks[8];
      const rawDist = Math.hypot(thumb.x - index.x, thumb.y - index.y);

      // Hysteresis: ON < 0.055, OFF > 0.075
      if (!ps.isPinching && rawDist < 0.055) {
        ps.isPinching = true;
      } else if (ps.isPinching && rawDist > 0.075) {
        ps.isPinching = false;
      }

      if (ps.isPinching) {
        ps.prevMidX = ps.midX;
        ps.prevMidY = ps.midY;
        ps.midX = (thumb.x + index.x) / 2;
        ps.midY = (thumb.y + index.y) / 2;
      }
    } else {
      ps.isPinching = false;
    }
  }

  const bothPinching = pinchState.hands[0].isPinching && pinchState.hands[1].isPinching;

  // 2-Hand Pinch-to-Zoom
  if (bothPinching) {
    const dist = Math.hypot(
      pinchState.hands[0].midX - pinchState.hands[1].midX,
      pinchState.hands[0].midY - pinchState.hands[1].midY
    );
    if (pinchState.prevTwoHandDist !== null) {
      const delta = pinchState.prevTwoHandDist - dist;
      earth.targetCamZ = Math.max(2.3, Math.min(6.5, earth.targetCamZ + delta * 8));
    }
    pinchState.prevTwoHandDist = dist;
  } else {
    pinchState.prevTwoHandDist = null;
  }

  // 1-Hand Pinch-to-Rotate (when not in 2-hand zoom)
  if (!bothPinching) {
    for (let h = 0; h < 2; h++) {
      const ps = pinchState.hands[h];
      if (ps.isPinching && ps.wasPinching) {
        const dX = ps.midX - ps.prevMidX;
        const dY = ps.midY - ps.prevMidY;
        earth.rotVelY += (state.mirrorMode ? -dX : dX) * 8;
        earth.rotVelX += dY * 8;
      }
    }
  }
}

function tickEarthGlobe(dt) {
  if (!earth.ready || !earth.earthGroup) return;

  const anyPinching = pinchState.hands[0].isPinching || pinchState.hands[1].isPinching;

  if (!anyPinching) {
    earth.rotVelX *= Math.pow(0.94, dt * 60);
    earth.rotVelY *= Math.pow(0.94, dt * 60);
    earth.rotVelY += 0.0012; // gentle auto-rotation
  }

  earth.earthGroup.rotation.y += earth.rotVelY * dt;
  earth.earthGroup.rotation.x += earth.rotVelX * dt;

  const maxTilt = Math.PI / 2.2;
  earth.earthGroup.rotation.x = Math.max(-maxTilt, Math.min(maxTilt, earth.earthGroup.rotation.x));

  const lerpZ = 1.0 - Math.exp(-8 * dt);
  earth.currentCamZ += (earth.targetCamZ - earth.currentCamZ) * lerpZ;
  earth.camera.position.z = earth.currentCamZ;

  if (earth.cloudMesh) {
    earth.cloudMesh.rotation.y += 0.00008;
  }
}

// ---------------------------------------------------------------------------
// 9. Coordinate Translation & Fingertip Rendering (Thumb & Index ONLY)
// ---------------------------------------------------------------------------
function toScreenX(normX) {
  return state.mirrorMode ? (1 - normX) * canvas.width : normX * canvas.width;
}

function toScreenY(normY) {
  return normY * canvas.height;
}

/**
 * Renders ONLY glowing circle points on Thumb (#4) and Index (#8) fingertips.
 * Skeletons, non-fingertip joints, middle/ring/pinky fingers, and UI badges are removed.
 */
function renderFingertipPoints() {
  for (let h = 0; h < 2; h++) {
    const slot = handSlots[h];
    if (slot.opacity <= 0.005 || !slot.renderedLandmarks) continue;

    ctx.save();
    ctx.globalAlpha = slot.opacity;

    for (const tipIdx of FINGERTIP_INDICES) {
      const pt = slot.renderedLandmarks[tipIdx];
      if (!pt) continue;

      const sx = toScreenX(pt.x);
      const sy = toScreenY(pt.y);

      // Outer glowing halo
      ctx.save();
      ctx.beginPath();
      ctx.arc(sx, sy, 9, 0, Math.PI * 2);
      ctx.strokeStyle = "rgba(255, 255, 255, 0.4)";
      ctx.lineWidth = 1.5;
      ctx.shadowColor = "rgba(255, 255, 255, 0.9)";
      ctx.shadowBlur = 10;
      ctx.stroke();

      // Inner crisp white core
      ctx.beginPath();
      ctx.arc(sx, sy, 5, 0, Math.PI * 2);
      ctx.fillStyle = "#ffffff";
      ctx.shadowColor = "rgba(255, 255, 255, 1.0)";
      ctx.shadowBlur = 8;
      ctx.fill();
      ctx.restore();
    }

    // Glowing energy ball between thumb and index fingertips when pinching
    const ps = pinchState.hands[h];
    if (ps.isPinching) {
      slot.pinchScale = Math.min(1.0, (slot.pinchScale || 0.0) + 0.15);
    } else {
      slot.pinchScale = Math.max(0.0, (slot.pinchScale || 0.0) - 0.15);
    }

    if (slot.pinchScale > 0.01) {
      const thumb = slot.renderedLandmarks[4];
      const index = slot.renderedLandmarks[8];
      const tx = toScreenX(thumb.x);
      const ty = toScreenY(thumb.y);
      const ix = toScreenX(index.x);
      const iy = toScreenY(index.y);

      // Center point right between the fingertips
      const mx = (tx + ix) / 2;
      const my = (ty + iy) / 2;

      // Pulse breathing animation
      const pulse = 1.0 + Math.sin(performance.now() * 0.01) * 0.12;
      const coreR = 5.5 * slot.pinchScale * pulse;
      const auraR = 15.0 * slot.pinchScale * pulse;

      ctx.save();
      ctx.globalAlpha = slot.opacity * slot.pinchScale;

      // Outer radial glow
      const grad = ctx.createRadialGradient(mx, my, 1, mx, my, auraR);
      grad.addColorStop(0.0, "rgba(255, 255, 255, 0.95)");
      grad.addColorStop(0.35, "rgba(56, 189, 248, 0.85)");
      grad.addColorStop(0.7, "rgba(14, 165, 233, 0.35)");
      grad.addColorStop(1.0, "rgba(14, 165, 233, 0.0)");

      ctx.beginPath();
      ctx.arc(mx, my, auraR, 0, Math.PI * 2);
      ctx.fillStyle = grad;
      ctx.shadowColor = "#38bdf8";
      ctx.shadowBlur = 18;
      ctx.fill();

      // Delicate outer energy ring
      ctx.beginPath();
      ctx.arc(mx, my, auraR * 0.78, 0, Math.PI * 2);
      ctx.strokeStyle = "rgba(255, 255, 255, 0.7)";
      ctx.lineWidth = 1.2;
      ctx.stroke();

      // Intense white core sphere
      ctx.beginPath();
      ctx.arc(mx, my, coreR, 0, Math.PI * 2);
      ctx.fillStyle = "#ffffff";
      ctx.shadowColor = "#ffffff";
      ctx.shadowBlur = 12;
      ctx.fill();

      ctx.restore();
    }

    ctx.restore();
  }
}

// ---------------------------------------------------------------------------
// 10. Real-Time Tracking & Decoupled Render Loop
// ---------------------------------------------------------------------------
function renderLoop(timestamp) {
  if (!state.webcamRunning) return;

  const nowSec = performance.now() / 1000;
  const dt = Math.min(0.1, Math.max(0.001, (timestamp - state.lastFrameTimestamp) / 1000));
  state.lastFrameTimestamp = timestamp;

  // 1. Camera Inference
  if (video.currentTime !== state.lastVideoTime && state.handLandmarker) {
    state.lastVideoTime = video.currentTime;

    let results = null;
    try {
      results = state.handLandmarker.detectForVideo(video, performance.now());
    } catch (e) {
      console.warn("HandLandmarker detection error:", e);
    }

    if (results && results.landmarks && results.landmarks.length > 0) {
      const detectedCount = results.landmarks.length;

      for (let h = 0; h < 2; h++) {
        const slot = handSlots[h];

        if (h < detectedCount) {
          const rawLandmarks = results.landmarks[h];

          if (nowSec - slot.lastDetectedTime > 0.150) {
            slot.resetFilters();
            slot.renderedLandmarks = null;
          }

          const filtered = [];
          for (let i = 0; i < 21; i++) {
            const pt = rawLandmarks[i];
            const fx = slot.filters[i].x.filter(pt.x, nowSec);
            const fy = slot.filters[i].y.filter(pt.y, nowSec);
            const fz = slot.filters[i].z.filter(pt.z || 0.0, nowSec);
            filtered.push({ x: fx, y: fy, z: fz });
          }

          slot.targetLandmarks = filtered;
          slot.lastDetectedTime = nowSec;
          slot.active = true;
        } else {
          slot.active = false;
        }
      }
    } else {
      for (let h = 0; h < 2; h++) {
        handSlots[h].active = false;
      }
    }
  }

  // 2. Sub-Frame Decoupled Lerp Interpolation
  const lerpFactor = Math.min(1.0, 1.0 - Math.exp(-36.0 * dt));
  const FADE_RATE = dt / 0.080;

  for (let h = 0; h < 2; h++) {
    const slot = handSlots[h];

    if (slot.active && nowSec - slot.lastDetectedTime <= 0.150) {
      slot.opacity = Math.min(1.0, slot.opacity + FADE_RATE);
    } else {
      slot.opacity = Math.max(0.0, slot.opacity - FADE_RATE);
      if (slot.opacity <= 0.001) {
        slot.targetLandmarks = null;
        slot.renderedLandmarks = null;
        slot.pinchScale = 0.0;
        slot.resetFilters();
      }
    }

    if (slot.targetLandmarks) {
      if (!slot.renderedLandmarks) {
        slot.renderedLandmarks = slot.targetLandmarks.map((p) => ({ ...p }));
      } else {
        for (let i = 0; i < 21; i++) {
          const r = slot.renderedLandmarks[i];
          const t = slot.targetLandmarks[i];
          r.x += (t.x - r.x) * lerpFactor;
          r.y += (t.y - r.y) * lerpFactor;
          r.z += (t.z - r.z) * lerpFactor;
        }
      }
    }
  }

  // 3. Clear 2D overlay canvas
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // 4. Update pinch gestures & render 3D Earth
  updatePinchGestures();
  if (earth.ready) {
    tickEarthGlobe(dt);
    earth.renderer.render(earth.scene, earth.camera);
  }

  // 5. Render ONLY Thumb (#4) and Index (#8) fingertip dots
  renderFingertipPoints();

  state.animationFrameId = requestAnimationFrame(renderLoop);
}

// ---------------------------------------------------------------------------
// 11. Camera & MediaPipe Initialization (100% Client-Side)
// ---------------------------------------------------------------------------
async function initMediaPipeVision() {
  try {
    let FilesetResolver, HandLandmarker;
    if (window.Vision) {
      FilesetResolver = window.Vision.FilesetResolver;
      HandLandmarker = window.Vision.HandLandmarker;
    } else {
      const m = await import("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/vision_bundle.js");
      FilesetResolver = m.FilesetResolver || window.Vision?.FilesetResolver;
      HandLandmarker = m.HandLandmarker || window.Vision?.HandLandmarker;
    }

    const vision = await FilesetResolver.forVisionTasks(WASM_CDN_URL);

    try {
      state.handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: MODEL_URL,
          delegate: "GPU",
        },
        runningMode: "VIDEO",
        numHands: 2,
        minHandDetectionConfidence: 0.5,
        minHandPresenceConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });
    } catch (gpuError) {
      console.warn("GPU delegate fallback to CPU:", gpuError);
      state.handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: MODEL_URL,
          delegate: "CPU",
        },
        runningMode: "VIDEO",
        numHands: 2,
        minHandDetectionConfidence: 0.5,
        minHandPresenceConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });
    }

    loadingOverlay?.classList.add("hidden");
    return true;
  } catch (err) {
    console.error("MediaPipe initialization error:", err);
    return false;
  }
}

async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 640 },
        height: { ideal: 480 },
        frameRate: { ideal: 60 },
      },
      audio: false,
    });

    state.stream = stream;
    video.srcObject = stream;

    await new Promise((resolve) => {
      video.onloadedmetadata = () => {
        video.play().then(resolve);
      };
    });

    resize();
    state.webcamRunning = true;
    state.lastVideoTime = -1;
    state.lastFrameTimestamp = performance.now();
    state.animationFrameId = requestAnimationFrame(renderLoop);
  } catch (err) {
    console.error("Camera access error:", err);
  }
}

// ---------------------------------------------------------------------------
// 12. Main Bootstrap
// ---------------------------------------------------------------------------
async function main() {
  resize();
  window.addEventListener("resize", resize);

  // Initialize Three.js Earth Globe with natural PBR satellite lighting
  initEarthGlobe().catch((e) => console.warn("Earth globe init error:", e));

  // Initialize MediaPipe vision and start background camera
  const visionReady = await initMediaPipeVision();
  if (visionReady) {
    await startCamera();
  }
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", main);
} else {
  main();
}
