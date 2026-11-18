import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(
    title="PyTracker - Real-time Hand & Gesture Tracking",
    description="Near-zero latency hand tracking server powered by FastAPI and MediaPipe Tasks Vision",
    version="1.0.0",
)

# Enable permissive CORS for seamless local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Point3D:
    """Represents a 3D point in normalized landmark space."""

    __slots__ = ("x", "y", "z")

    def __init__(self, x: float, y: float, z: float = 0.0) -> None:
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def to_dict(self) -> Dict[str, float]:
        return {"x": round(self.x, 4), "y": round(self.y, 4), "z": round(self.z, 4)}


class GestureAnalyzer:
    """
    Robust real-time hand gesture analyzer using 3D skeletal geometry.
    Computes individual finger extension/curl states, normalized pinch metrics,
    and classifies gestures with rotation and scale invariance.
    """

    # MediaPipe Hand Landmark Indices:
    # 0: Wrist
    # Thumb: 1: CMC, 2: MCP, 3: IP, 4: TIP
    # Index: 5: MCP, 6: PIP, 7: DIP, 8: TIP
    # Middle: 9: MCP, 10: PIP, 11: DIP, 12: TIP
    # Ring: 13: MCP, 14: PIP, 15: DIP, 16: TIP
    # Pinky: 17: MCP, 18: PIP, 19: DIP, 20: TIP

    FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")

    # Joint sequences for 4 non-thumb fingers: (MCP, PIP, DIP, TIP)
    FINGER_JOINTS = {
        "index": (5, 6, 7, 8),
        "middle": (9, 10, 11, 12),
        "ring": (13, 14, 15, 16),
        "pinky": (17, 18, 19, 20),
    }

    GESTURE_METADATA = {
        "OPEN_PALM": {"display": "Open Palm", "emoji": "🖐️", "description": "All fingers extended"},
        "FIST": {"display": "Fist", "emoji": "✊", "description": "All fingers curled"},
        "POINTING": {"display": "Pointing", "emoji": "👆", "description": "Index extended, others curled"},
        "PEACE": {"display": "Peace / Victory", "emoji": "✌️", "description": "Index & Middle extended"},
        "PINCH": {"display": "Pinch", "emoji": "🤏", "description": "Thumb & Index tips touching"},
        "OK_SIGN": {"display": "OK Sign", "emoji": "👌", "description": "Pinch with other fingers open"},
        "THUMBS_UP": {"display": "Thumbs Up", "emoji": "👍", "description": "Thumb up, others curled"},
        "THUMBS_DOWN": {"display": "Thumbs Down", "emoji": "👎", "description": "Thumb down, others curled"},
        "ROCK": {"display": "Rock On", "emoji": "🤘", "description": "Index & Pinky extended"},
        "CALL_ME": {"display": "Call Me", "emoji": "🤙", "description": "Thumb & Pinky extended"},
        "GUN": {"display": "Hand Gun", "emoji": "👉", "description": "Thumb & Index extended"},
        "THREE": {"display": "Three", "emoji": "3️⃣", "description": "Index, Middle & Ring extended"},
        "FOUR": {"display": "Four", "emoji": "4️⃣", "description": "Four fingers extended, thumb curled"},
        "UNKNOWN": {"display": "Tracking Hand", "emoji": "✋", "description": "Interpreting gesture..."},
    }

    @staticmethod
    def euclidean_distance(p1: Point3D, p2: Point3D) -> float:
        """Calculate Euclidean distance between two 3D points."""
        dx = p1.x - p2.x
        dy = p1.y - p2.y
        dz = p1.z - p2.z
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    @staticmethod
    def joint_angle(p1: Point3D, vertex: Point3D, p2: Point3D) -> float:
        """
        Calculate angle in degrees at vertex formed by rays (vertex -> p1) and (vertex -> p2).
        Returns value in [0.0, 180.0].
        """
        v1x, v1y, v1z = p1.x - vertex.x, p1.y - vertex.y, p1.z - vertex.z
        v2x, v2y, v2z = p2.x - vertex.x, p2.y - vertex.y, p2.z - vertex.z

        norm1 = math.sqrt(v1x * v1x + v1y * v1y + v1z * v1z)
        norm2 = math.sqrt(v2x * v2x + v2y * v2y + v2z * v2z)

        if norm1 < 1e-6 or norm2 < 1e-6:
            return 0.0

        dot = v1x * v2x + v1y * v2y + v1z * v2z
        cos_theta = max(-1.0, min(1.0, dot / (norm1 * norm2)))
        return math.degrees(math.acos(cos_theta))

    @staticmethod
    def clamp(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
        return max(min_val, min(max_val, value))

    def analyze_finger(
        self,
        landmarks: List[Point3D],
        finger_name: str,
        palm_scale: float,
    ) -> Dict[str, Any]:
        """
        Determine if an individual non-thumb finger is extended vs. curled.
        Uses a combination of tip-to-MCP segment extension ratio, tip-to-wrist distance,
        and PIP joint flexion angle.
        """
        mcp_idx, pip_idx, dip_idx, tip_idx = self.FINGER_JOINTS[finger_name]
        wrist = landmarks[0]
        mcp = landmarks[mcp_idx]
        pip = landmarks[pip_idx]
        dip = landmarks[dip_idx]
        tip = landmarks[tip_idx]

        # Segment lengths along the finger
        seg1 = self.euclidean_distance(mcp, pip)
        seg2 = self.euclidean_distance(pip, dip)
        seg3 = self.euclidean_distance(dip, tip)
        total_len = seg1 + seg2 + seg3

        if total_len < 1e-5:
            return {"extended": False, "score": 0.0, "angle": 0.0}

        # Tip-to-MCP direct distance vs. total finger length
        direct_len = self.euclidean_distance(tip, mcp)
        extension_ratio = direct_len / total_len

        # Distances relative to wrist
        d_tip_wrist = self.euclidean_distance(tip, wrist)
        d_pip_wrist = self.euclidean_distance(pip, wrist)
        wrist_ratio = d_tip_wrist / max(1e-5, d_pip_wrist)

        # Flexion angle at PIP joint (straight is near 180 degrees)
        angle = self.joint_angle(mcp, pip, tip)

        # Continuous extension score [0.0, 1.0]
        # Fully straight: ratio ~ 0.95-1.0, angle ~ 160-180 deg
        # Curled fist: ratio ~ 0.3-0.45, angle < 90 deg
        norm_ratio = self.clamp((extension_ratio - 0.40) / 0.45)
        norm_angle = self.clamp((angle - 80.0) / 80.0)
        norm_wrist = self.clamp((wrist_ratio - 0.95) / 0.35)

        combined_score = 0.50 * norm_ratio + 0.30 * norm_angle + 0.20 * norm_wrist
        combined_score = self.clamp(combined_score)

        # Finger is considered extended if score exceeds threshold
        is_extended = combined_score >= 0.58 and d_tip_wrist > d_pip_wrist

        return {
            "extended": bool(is_extended),
            "score": round(combined_score, 3),
            "angle": round(angle, 1),
            "extension_ratio": round(extension_ratio, 3),
        }

    def analyze_thumb(
        self,
        landmarks: List[Point3D],
        handedness: str,
        palm_scale: float,
    ) -> Dict[str, Any]:
        """
        Analyze thumb extension vs. curled state.
        Thumb articulates differently than fingers: it opposes the palm.
        We evaluate tip distance to CMC (1), distance to Pinky MCP (17),
        distance to Index MCP (5), and thumb IP joint angle.
        """
        wrist = landmarks[0]
        cmc = landmarks[1]
        mcp = landmarks[2]
        ip = landmarks[3]
        tip = landmarks[4]
        index_mcp = landmarks[5]
        pinky_mcp = landmarks[17]

        # Total thumb bone length
        seg1 = self.euclidean_distance(cmc, mcp)
        seg2 = self.euclidean_distance(mcp, ip)
        seg3 = self.euclidean_distance(ip, tip)
        total_thumb_len = seg1 + seg2 + seg3

        if total_thumb_len < 1e-5:
            return {"extended": False, "score": 0.0, "angle": 0.0}

        # Distance from thumb tip to CMC
        d_tip_cmc = self.euclidean_distance(tip, cmc)
        straight_ratio = d_tip_cmc / total_thumb_len

        # Distance from thumb tip to Pinky MCP (opposing base)
        d_tip_pinky = self.euclidean_distance(tip, pinky_mcp)
        d_mcp_pinky = self.euclidean_distance(mcp, pinky_mcp)
        pinky_span_ratio = d_tip_pinky / max(1e-5, d_mcp_pinky)

        # Distance from thumb tip to Index MCP
        d_tip_index_mcp = self.euclidean_distance(tip, index_mcp)
        index_span_norm = d_tip_index_mcp / max(1e-5, palm_scale)

        # Joint angle at thumb IP joint
        angle = self.joint_angle(mcp, ip, tip)

        # Compute continuous thumb extension score
        norm_straight = self.clamp((straight_ratio - 0.50) / 0.40)
        norm_pinky = self.clamp((pinky_span_ratio - 0.85) / 0.55)
        norm_angle = self.clamp((angle - 110.0) / 60.0)

        thumb_score = 0.45 * norm_straight + 0.35 * norm_pinky + 0.20 * norm_angle
        thumb_score = self.clamp(thumb_score)

        # Thumb extended criteria
        is_extended = thumb_score >= 0.55 and pinky_span_ratio > 1.05

        return {
            "extended": bool(is_extended),
            "score": round(thumb_score, 3),
            "angle": round(angle, 1),
            "span_ratio": round(pinky_span_ratio, 3),
        }

    def analyze_hand(self, raw_landmarks: List[Dict[str, float]], handedness: str = "Right") -> Dict[str, Any]:
        """
        Analyze all 21 landmarks of a detected hand.
        Computes finger states, pinch metrics, and recognized gesture.
        """
        if len(raw_landmarks) < 21:
            return {
                "handedness": handedness,
                "gesture": "UNKNOWN",
                "gesture_display": self.GESTURE_METADATA["UNKNOWN"]["display"],
                "gesture_emoji": self.GESTURE_METADATA["UNKNOWN"]["emoji"],
                "confidence": 0.0,
                "is_pinch": False,
                "pinch_distance": 1.0,
                "pinch_strength": 0.0,
                "fingers": {},
            }

        landmarks = [Point3D(pt.get("x", 0.0), pt.get("y", 0.0), pt.get("z", 0.0)) for pt in raw_landmarks]

        wrist = landmarks[0]
        middle_mcp = landmarks[9]
        index_mcp = landmarks[5]
        pinky_mcp = landmarks[17]

        # Characteristic palm scale: distance between wrist (0) and middle MCP (9)
        palm_scale = self.euclidean_distance(wrist, middle_mcp)
        if palm_scale < 1e-4:
            palm_scale = max(1e-4, self.euclidean_distance(index_mcp, pinky_mcp))

        # Analyze 5 individual fingers
        fingers_state: Dict[str, Dict[str, Any]] = {}

        # 1. Thumb
        fingers_state["thumb"] = self.analyze_thumb(landmarks, handedness, palm_scale)

        # 2. Index, Middle, Ring, Pinky
        for name in ("index", "middle", "ring", "pinky"):
            fingers_state[name] = self.analyze_finger(landmarks, name, palm_scale)

        # Normalized pinch metrics between Thumb tip (4) and Index tip (8)
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]

        raw_pinch_dist = self.euclidean_distance(thumb_tip, index_tip)
        norm_pinch_dist = raw_pinch_dist / max(1e-4, palm_scale)

        # Pinch threshold: normalized distance < 0.28 represents finger touching
        PINCH_THRESHOLD = 0.28
        is_pinch = norm_pinch_dist <= PINCH_THRESHOLD
        pinch_strength = self.clamp(1.0 - (norm_pinch_dist / PINCH_THRESHOLD))

        # Check secondary pinch with middle finger
        raw_middle_pinch = self.euclidean_distance(thumb_tip, middle_tip)
        norm_middle_pinch = raw_middle_pinch / max(1e-4, palm_scale)

        # Boolean finger extension flags
        t_ext = fingers_state["thumb"]["extended"]
        i_ext = fingers_state["index"]["extended"]
        m_ext = fingers_state["middle"]["extended"]
        r_ext = fingers_state["ring"]["extended"]
        p_ext = fingers_state["pinky"]["extended"]

        # Thumb vertical direction for thumbs up/down
        # In MediaPipe screen coords: y=0 is top, y=1 is bottom
        thumb_mcp = landmarks[2]
        thumb_dy = thumb_tip.y - thumb_mcp.y  # negative when pointing upwards
        thumb_wrist_dy = thumb_tip.y - wrist.y

        # Gesture Classification Pipeline
        gesture = "UNKNOWN"
        confidence = 0.85

        # 1. OK SIGN: Thumb & Index pinching, while Middle, Ring, Pinky extended
        if is_pinch and m_ext and r_ext:
            gesture = "OK_SIGN"
            confidence = 0.95

        # 2. PINCH: Thumb & Index touching, index not fully extended straight
        elif is_pinch and not (m_ext and r_ext and p_ext):
            gesture = "PINCH"
            confidence = 0.90 + 0.10 * pinch_strength

        # 3. OPEN PALM: All 5 fingers extended
        elif t_ext and i_ext and m_ext and r_ext and p_ext:
            gesture = "OPEN_PALM"
            avg_score = sum(fingers_state[f]["score"] for f in self.FINGER_NAMES) / 5.0
            confidence = 0.88 + 0.12 * avg_score

        # 4. FIST: All 5 fingers curled
        elif not t_ext and not i_ext and not m_ext and not r_ext and not p_ext:
            gesture = "FIST"
            inv_avg = 1.0 - (sum(fingers_state[f]["score"] for f in self.FINGER_NAMES) / 5.0)
            confidence = 0.85 + 0.15 * inv_avg

        # 5. POINTING: Only index finger extended
        elif i_ext and not m_ext and not r_ext and not p_ext:
            if t_ext and (thumb_dy < -0.05 or abs(thumb_tip.x - index_mcp.x) > 0.1):
                # Thumb pointing out + index extended -> Hand Gun
                gesture = "GUN"
                confidence = 0.92
            else:
                gesture = "POINTING"
                confidence = 0.95

        # 6. PEACE / VICTORY: Index and Middle extended, Ring and Pinky curled
        elif i_ext and m_ext and not r_ext and not p_ext:
            gesture = "PEACE"
            confidence = 0.94

        # 7. ROCK ON: Index and Pinky extended, Middle and Ring curled
        elif i_ext and p_ext and not m_ext and not r_ext:
            gesture = "ROCK"
            confidence = 0.93

        # 8. CALL ME: Thumb and Pinky extended, others curled
        elif t_ext and p_ext and not i_ext and not m_ext and not r_ext:
            gesture = "CALL_ME"
            confidence = 0.92

        # 9. THUMBS UP / DOWN: Thumb extended, all other 4 curled
        elif t_ext and not i_ext and not m_ext and not r_ext and not p_ext:
            if thumb_dy < -0.04 and thumb_wrist_dy < -0.08:
                gesture = "THUMBS_UP"
                confidence = 0.94
            elif thumb_dy > 0.04 and thumb_wrist_dy > 0.08:
                gesture = "THUMBS_DOWN"
                confidence = 0.93
            else:
                gesture = "THUMBS_UP"
                confidence = 0.85

        # 10. THREE: Index, Middle, Ring extended, Pinky curled
        elif i_ext and m_ext and r_ext and not p_ext:
            gesture = "THREE"
            confidence = 0.90

        # 11. FOUR: Four fingers extended, Thumb curled
        elif not t_ext and i_ext and m_ext and r_ext and p_ext:
            gesture = "FOUR"
            confidence = 0.91

        meta = self.GESTURE_METADATA.get(gesture, self.GESTURE_METADATA["UNKNOWN"])

        # Compute palm bounding box and centroid for spatial reference
        xs = [pt.x for pt in landmarks]
        ys = [pt.y for pt in landmarks]
        bbox = {
            "x_min": round(min(xs), 4),
            "y_min": round(min(ys), 4),
            "x_max": round(max(xs), 4),
            "y_max": round(max(ys), 4),
        }
        centroid = {
            "x": round((wrist.x + middle_mcp.x) / 2.0, 4),
            "y": round((wrist.y + middle_mcp.y) / 2.0, 4),
        }

        return {
            "handedness": handedness,
            "gesture": gesture,
            "gesture_display": meta["display"],
            "gesture_emoji": meta["emoji"],
            "description": meta["description"],
            "confidence": round(confidence, 3),
            "is_pinch": bool(is_pinch),
            "pinch_distance": round(norm_pinch_dist, 3),
            "pinch_strength": round(pinch_strength, 3),
            "palm_scale": round(palm_scale, 4),
            "centroid": centroid,
            "bbox": bbox,
            "fingers": {
                name: {
                    "extended": fingers_state[name]["extended"],
                    "score": fingers_state[name]["score"],
                    "angle": fingers_state[name]["angle"],
                }
                for name in self.FINGER_NAMES
            },
        }


# Singleton gesture analyzer instance
analyzer = GestureAnalyzer()

# Determine project paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")


@app.get("/")
async def serve_index():
    """Serve the single-page application frontend."""
    index_file = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "Frontend index.html not found", "path": index_file}


@app.websocket("/ws/landmarks")
async def websocket_landmarks(websocket: WebSocket):
    """
    Persistent, non-blocking WebSocket endpoint for receiving 3D landmarks
    and streaming back low-latency gesture recognition analysis.
    """
    await websocket.accept()
    client_host = websocket.client.host if websocket.client else "unknown"

    try:
        while True:
            # Receive client payload: { client_time, hands: [ { handedness, landmarks: [...] } ] }
            data = await websocket.receive_json()
            t0 = time.perf_counter()

            client_time = data.get("client_time", 0.0)
            raw_hands = data.get("hands", [])

            analyzed_hands = []
            for hand_obj in raw_hands:
                handedness = hand_obj.get("handedness", "Right")
                landmarks = hand_obj.get("landmarks", [])
                analysis = analyzer.analyze_hand(landmarks, handedness=handedness)
                analyzed_hands.append(analysis)

            t1 = time.perf_counter()
            process_time_ms = (t1 - t0) * 1000.0

            # Lightweight response payload
            response = {
                "client_time": client_time,
                "server_time": time.time(),
                "process_time_ms": round(process_time_ms, 3),
                "hands": analyzed_hands,
                "hands_count": len(analyzed_hands),
            }

            await websocket.send_json(response)

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        # Handle unexpected socket error gracefully
        try:
            await websocket.close()
        except Exception:
            pass


# Mount static files for assets (CSS, JS, icons)
if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
    # Also mount root static fallback so /app.js and /style.css load seamlessly
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend_root")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
