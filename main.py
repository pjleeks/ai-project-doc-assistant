"""
OrbitNet V6 Agent - Standalone Production Agent for Orbit Wars Kaggle Competition
================================================================
Complete self-contained agent with:
- Optimized ONNX model inference
- Physics calculations (orbital mechanics, fleet speed, collision detection)
- 18-feature decision making
- Batch scoring for efficiency
- Defensive & offensive strategies
"""

import os
import sys
import math
import json
import numpy as np
from collections import namedtuple
from functools import lru_cache
from typing import List, Tuple, Optional, Dict

# Try to import ONNX Runtime, fallback if not available
try:
    import onnxruntime as rt
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("⚠️ Warning: onnxruntime not available. Agent will use heuristics only.")

# ================================================================
# CONSTANTS & CONFIGURATION
# ================================================================
CENTER_X, CENTER_Y = 50.0, 50.0
BOARD_SIZE = 100.0
SUN_R, SUN_SAFETY = 10.0, 0.5
MAX_SPEED = 6.0
MIN_NN_SCORE = 0.35  # ONNX confidence threshold

# Danger zones discovered from top-10 episodes
TRAPS = [
    (15, 83), (5, 67), (9, 76), (75, 75), (89, 61),
    (6, 28), (90, 68), (95, 69), (56, 9), (36, 18)
]

# Data structures
Planet = namedtuple("Planet", ["id", "owner", "x", "y", "radius", "ships", "production"])
Fleet = namedtuple("Fleet", ["id", "owner", "x", "y", "angle", "from_planet_id", "ships"])

# ================================================================
# ONNX MODEL MANAGEMENT (Lazy Loading)
# ================================================================
_onnx_session = None
_input_name = None
_output_name = None

def get_onnx_session():
    """Lazy-load ONNX session once per game"""
    global _onnx_session, _input_name, _output_name
    
    if _onnx_session is not None:
        return _onnx_session
    
    if not ONNX_AVAILABLE:
        return None
    
    try:
        # Try multiple possible paths
        possible_paths = [
            "orbit_model_v6.onnx",
            "orbit_model.onnx",
            "/kaggle/working/orbit_model_v6.onnx",
            "/kaggle/working/orbit_model.onnx",
            "./orbit_model_v6.onnx",
            "./orbit_model.onnx",
        ]
        
        model_path = None
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if model_path is None:
            print(f"⚠️ ONNX model not found. Searched: {possible_paths}")
            return None
        
        # Initialize session with optimizations
        opts = rt.SessionOptions()
        opts.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 4
        
        _onnx_session = rt.InferenceSession(
            model_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"]
        )
        
        _input_name = _onnx_session.get_inputs()[0].name
        _output_name = _onnx_session.get_outputs()[0].name
        
        print(f"✅ ONNX model loaded from: {model_path}")
        return _onnx_session
    
    except Exception as e:
        print(f"❌ Failed to load ONNX model: {e}")
        return None

# ================================================================
# FEATURE ENGINEERING & HEATMAP FUNCTIONS
# ================================================================
def get_danger_heat(tx: float, ty: float) -> float:
    """Distance-based danger heat from known trap zones [0, 1]"""
    if not TRAPS:
        return 0.0
    min_d = min(math.hypot(tx - t[0], ty - t[1]) for t in TRAPS)
    return max(0.0, min(1.0, 1.0 - (min_d / 15.0)))

@lru_cache(maxsize=512)
def get_danger_heat_cached(tx_rounded: int, ty_rounded: int) -> float:
    """Cached version for integer coordinates"""
    return get_danger_heat(float(tx_rounded), float(ty_rounded))

def min_mirror_dist(tx: float, ty: float, sx: float, sy: float) -> float:
    """Distance to closest mirrored variant (board symmetry)"""
    d1 = math.hypot(tx - (BOARD_SIZE - sx), ty - sy)
    d2 = math.hypot(tx - sx, ty - (BOARD_SIZE - sy))
    d3 = math.hypot(tx - (BOARD_SIZE - sx), ty - (BOARD_SIZE - sy))
    return min(d1, d2, d3)

# ================================================================
# PHYSICS CALCULATIONS
# ================================================================
@lru_cache(maxsize=256)
def fleet_speed(ships: int) -> float:
    """Calculate fleet speed based on ship count (logarithmic curve)"""
    if ships <= 1:
        return 1.0
    log_factor = min(1.0, math.log(ships) / math.log(1000.0))
    return 1.0 + (MAX_SPEED - 1.0) * (log_factor ** 1.5)

def get_target_pos(
    tgt: Planet,
    turns: int,
    ang_vel: float,
    comets: List[Dict],
    comet_ids: set
) -> Tuple[float, float]:
    """Predict target position accounting for orbital motion and comets"""
    
    # Handle comets (follow predefined paths)
    if tgt.id in comet_ids:
        for c in comets:
            if tgt.id in c.get("planet_ids", []):
                try:
                    idx = c["planet_ids"].index(tgt.id)
                    f_idx = c.get("path_index", 0) + turns
                    if idx < len(c["paths"]) and 0 <= f_idx < len(c["paths"][idx]):
                        return tuple(c["paths"][idx][f_idx])
                except (IndexError, KeyError, ValueError):
                    pass
        return (tgt.x, tgt.y)
    
    # Handle orbiting planets
    if ang_vel == 0.0:
        return (tgt.x, tgt.y)
    
    r = math.hypot(tgt.x - CENTER_X, tgt.y - CENTER_Y)
    ang = math.atan2(tgt.y - CENTER_Y, tgt.x - CENTER_X) + ang_vel * turns
    return (CENTER_X + r * math.cos(ang), CENTER_Y + r * math.sin(ang))

def point_to_segment_dist(
    px: float, py: float,
    x1: float, y1: float,
    x2: float, y2: float
) -> float:
    """Distance from point to line segment (for sun collision detection)"""
    dx, dy = x2 - x1, y2 - y1
    l2 = dx * dx + dy * dy
    
    if l2 == 0:
        return math.hypot(px - x1, py - y1)
    
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / l2))
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy
    
    return math.hypot(px - closest_x, py - closest_y)

def plan_flight(
    src: Planet,
    tgt: Planet,
    ships: int,
    ang_vel: float,
    comets: List[Dict],
    comet_ids: set
) -> Tuple[Optional[float], int, float, float, float]:
    """
    Plan intercept trajectory for fleet
    Returns: (angle, eta_turns, target_x, target_y, speed) or (None, 999, ...) if blocked
    """
    speed = fleet_speed(ships)
    tx, ty = tgt.x, tgt.y
    eta = 0.0
    
    # Fixed-point iteration to converge on moving target
    for _ in range(8):
        dist = math.hypot(tx - src.x, ty - src.y)
        flight_dist = max(0.0, dist - src.radius - tgt.radius - 0.1)
        eta = flight_dist / speed if speed > 0 else 0.0
        tx, ty = get_target_pos(tgt, int(math.ceil(eta)), ang_vel, comets, comet_ids)
    
    angle = math.atan2(ty - src.y, tx - src.x)
    
    # Collision check with sun
    sx = src.x + math.cos(angle) * (src.radius + 0.1)
    sy = src.y + math.sin(angle) * (src.radius + 0.1)
    ex = sx + math.cos(angle) * (eta * speed)
    ey = sy + math.sin(angle) * (eta * speed)
    
    sun_dist = point_to_segment_dist(CENTER_X, CENTER_Y, sx, sy, ex, ey)
    if sun_dist <= SUN_R + SUN_SAFETY:
        return None, 999, tx, ty, speed
    
    return angle, int(math.ceil(eta)), tx, ty, speed

# ================================================================
# ONNX SCORING (Batch Inference)
# ================================================================
def extract_features(
    src: Planet,
    tgt: Planet,
    eta: int,
    speed: float,
    tx: float, ty: float,
    threat_level: float,
    player_ships: int,
    player_territory: int,
    step: int,
    is_defensive: bool
) -> np.ndarray:
    """Extract 18-feature vector for ONNX model"""
    
    # Normalize features to [0, 1] range
    features = np.array([
        abs(tx - 50.0) / 50.0,                                      # 1. Folded X
        abs(ty - 50.0) / 50.0,                                      # 2. Folded Y
        tgt.production / 5.0,                                        # 3. Target production
        min(1.0, tgt.ships / 100.0),                                 # 4. Target garrison
        1.0 if (src.ships - eta * speed / 2) >= 20 else 0.0,       # 5. Is significant
        min(1.0, eta / 50.0),                                        # 6. Normalized ETA
        get_danger_heat(tx, ty),                                     # 7. Danger heat
        min(1.0, eta / max(1.0, src.ships)),                        # 8. Capital risk
        min(1.0, tgt.production / max(1.0, eta * speed / 2)),       # 9. ROI
        tgt.production / 5.0,                                        # 10. Income gain
        1.0 if is_defensive else 0.0,                               # 11. Is defensive
        1.0 if tgt.owner == -1 else (0.5 if tgt.owner == src.owner else -0.5),  # 12. Owner type
        min(1.0, player_ships / 500.0),                             # 13. Player fleet size
        min(1.0, player_territory / 40.0),                          # 14. Territory control
        min(1.0, step / 500.0),                                     # 15. Game phase
        min(1.0, min([math.hypot(tx - p[0], ty - p[1]) / 50.0 for p in TRAPS] + [1.0])),  # 16. Trap proximity
        min(1.0, threat_level / 100.0),                             # 17. Threat level
        min(1.0, math.log(max(1, eta * int(speed)) + 1) / math.log(1000)),  # 18. Log fleet size
    ], dtype=np.float32)
    
    return features

def batch_score_moves(
    candidates: List[Dict],
    session
) -> List[Dict]:
    """Score multiple moves in single ONNX batch call"""
    
    if not session or len(candidates) == 0:
        # Fallback: score by heuristics
        for candidate in candidates:
            candidate['onnx_score'] = 0.5  # Neutral score
        return candidates
    
    try:
        # Stack all features
        features_list = [c['features'] for c in candidates]
        batch_input = np.vstack(features_list) if features_list else np.empty((0, 18))
        
        # Get input/output names
        input_name = _input_name
        output_name = _output_name
        
        # Single batch inference
        if len(batch_input) > 0:
            scores = session.run([output_name], {input_name: batch_input})[0]
            for i, candidate in enumerate(candidates):
                candidate['onnx_score'] = float(scores[i][0])
        else:
            for candidate in candidates:
                candidate['onnx_score'] = 0.5
    
    except Exception as e:
        print(f"⚠️ Batch scoring error: {e}")
        for candidate in candidates:
            candidate['onnx_score'] = 0.5
    
    return candidates

# ================================================================
# THREAT ASSESSMENT
# ================================================================
def compute_threat_levels(
    obs: Dict,
    my_planets: List[Planet],
    player_id: int
) -> Dict[int, float]:
    """Compute incoming threat to each planet"""
    
    threat_map = {p.id: 0.0 for p in my_planets}
    fleets = [Fleet(*f) for f in obs.get("fleets", [])]
    
    for fleet in fleets:
        if fleet.owner == player_id:
            continue  # Ignore own fleets
        
        # Find which planet this fleet threatens
        for planet in my_planets:
            dist = math.hypot(fleet.x - planet.x, fleet.y - planet.y)
            if dist < 50:  # Within engagement range
                threat_map[planet.id] += fleet.ships / max(1.0, dist)
    
    return threat_map

# ================================================================
# MAIN AGENT FUNCTION
# ================================================================
def agent(obs: Dict, config=None):
    """
    Main agent function for Kaggle Orbit Wars competition
    
    Args:
        obs: Observation dict with planets, fleets, angular_velocity, etc.
        config: Competition configuration (unused)
    
    Returns:
        List of moves: [[from_planet_id, angle_radians, num_ships], ...]
    """
    
    # Parse observation
    player_id = obs.get("player", 0)
    planets = [Planet(*p) for p in obs.get("planets", [])]
    fleets = [Fleet(*f) for f in obs.get("fleets", [])]
    ang_vel = obs.get("angular_velocity", 0.0)
    comets = obs.get("comets", [])
    comet_ids = set(obs.get("comet_planet_ids", []))
    step = obs.get("step", 0)
    
    # Get ONNX session
    session = get_onnx_session() if ONNX_AVAILABLE else None
    
    # Identify my planets
    my_planets = [p for p in planets if p.owner == player_id]
    if not my_planets:
        return []
    
    # Calculate threat levels
    threat_map = compute_threat_levels(obs, my_planets, player_id)
    
    # Compute aggregate stats
    my_ships = sum(p.ships for p in my_planets)
    my_territory = len(my_planets)
    enemy_ships = sum(p.ships for p in planets if p.owner != -1 and p.owner != player_id)
    
    # Generate candidates
    candidates = []
    
    for src in my_planets:
        if src.ships <= 1:
            continue
        
        available = src.ships
        
        for tgt in planets:
            if tgt.id == src.id:
                continue
            
            is_defensive = (tgt.owner == player_id)
            
            # Plan flight
            angle, eta, tx, ty, speed = plan_flight(src, tgt, available, ang_vel, comets, comet_ids)
            if angle is None:
                continue
            
            # Determine deployment size
            if is_defensive:
                # Defend against threats
                needed = max(1, int(threat_map.get(tgt.id, 0) * 1.1))
                ships_to_send = min(available, needed)
            else:
                # Offensive: estimate enemy strength
                enemy_future = tgt.ships + (tgt.production * eta if tgt.owner != -1 else 0)
                ships_to_send = int(enemy_future * 1.2) + 1
            
            # Filter by budget
            if ships_to_send > available or ships_to_send < 1:
                continue
            
            # Extract features
            features = extract_features(
                src, tgt, eta, speed, tx, ty,
                threat_map.get(src.id, 0),
                my_ships, my_territory, step,
                is_defensive
            )
            
            candidates.append({
                'src_id': src.id,
                'angle': angle,
                'ships': ships_to_send,
                'target_id': tgt.id,
                'features': features,
                'is_defensive': is_defensive,
                'onnx_score': 0.5  # Default
            })
    
    # Score candidates with ONNX
    if session and len(candidates) > 0:
        candidates = batch_score_moves(candidates, session)
    
    # Filter by confidence threshold
    candidates = [c for c in candidates if c['onnx_score'] >= MIN_NN_SCORE]
    
    # Sort by score (highest first)
    candidates.sort(key=lambda x: x['onnx_score'], reverse=True)
    
    # Convert to moves, respecting ship budget
    moves = []
    ships_deployed = {p.id: 0 for p in my_planets}
    
    for candidate in candidates[:50]:  # Max 50 moves
        src_id = candidate['src_id']
        ships_available = next(p.ships for p in my_planets if p.id == src_id)
        ships_to_deploy = candidate['ships']
        
        if ships_deployed[src_id] + ships_to_deploy <= ships_available:
            moves.append([
                src_id,
                candidate['angle'],
                ships_to_deploy
            ])
            ships_deployed[src_id] += ships_to_deploy
    
    return moves

# ================================================================
# TESTING & DEBUGGING
# ================================================================
if __name__ == "__main__":
    # Test with dummy observation
    dummy_obs = {
        "player": 0,
        "planets": [
            [0, 0, 25.0, 25.0, 2.0, 10, 2],
            [1, -1, 75.0, 75.0, 1.5, 20, 1],
            [2, 1, 50.0, 25.0, 1.0, 15, 1],
        ],
        "fleets": [],
        "angular_velocity": 0.025,
        "comets": [],
        "comet_planet_ids": [],
        "step": 10,
    }
    
    print("🚀 Testing OrbitNet V6 Agent...")
    print("=" * 60)
    
    result = agent(dummy_obs)
    print(f"\n📋 Agent returned {len(result)} moves:")
    for i, move in enumerate(result[:5]):
        print(f"   Move {i+1}: Planet {move[0]} → Angle {move[1]:.3f} rad, Ships {move[2]}")
    
    print("\n✅ Agent test complete!")
