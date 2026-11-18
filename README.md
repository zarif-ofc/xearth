# xearth 🌍✨

**Interactive 3D Earth Globe controlled via real-time hand pinch gestures and computer vision**, running 100% client-side in the browser on WebAssembly & WebGL/GPU.

---

## 🌟 Highlights

- **Pure Client-Side Inference:** MediaPipe HandLandmarker (WASM/GPU) runs locally at 60 FPS in your browser with zero latency.
- **Interactive 3D Earth:** High-fidelity Three.js Earth with custom GPU-graded vibrant oceans, lush emerald continents, soft satellite lighting, atmospheric rim glow, and drifting clouds.
- **Natural Pinch Controls:**
  - **1-Hand Pinch-to-Rotate:** Pinch your thumb and index fingertips together to rotate the globe with smooth momentum damping and inertia physics.
  - **2-Hand Pinch-to-Zoom:** Pinch with both hands to fluidly zoom in and out.
  - **Glowing Energy Orb:** A small, pulsating energy ball ignites directly between your fingertips whenever you pinch.
- **1€ Adaptive Anti-Jitter Filter:** Jitter-free stationary tracking with instantaneous responsiveness during fast hand movements.
- **Minimalist Aesthetic:** Clean, full-screen holographic view set against a pure black background.

---

## 🚀 Quickstart

Run directly using any static HTTP server (no build step or backend required):

```bash
# Python built-in static server
python -m http.server 8000 --directory frontend
```

Or using Node's `serve` / `npx`:

```bash
npx serve frontend
```

Then open `http://localhost:8000` in your browser and allow webcam access.

---

## 📂 Project Structure

```text
xearth/
├── frontend/
│   ├── index.html       # Minimal full-screen viewport layout
│   ├── app.js           # MediaPipe vision, 1€ filter, Three.js 3D Earth & pinch controls
│   └── style.css        # Pure black full-screen styling
├── backend/             # Optional FastAPI server & gesture analyzer
│   ├── __init__.py
│   └── main.py
├── requirements.txt     # Optional Python dependencies
├── .gitignore
└── README.md
```

---

## 📜 License

MIT License. Built with MediaPipe Tasks Vision and Three.js.
