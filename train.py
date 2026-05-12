"""
OrbitNet V6 Training Pipeline - Standalone Training Script
================================================================
Complete self-contained training with:
- Improved feature engineering (18 features vs 12)
- Batch normalization & dropout regularization
- Focal loss for imbalanced data
- Learning rate scheduling
- Complete error handling
"""

import os
import sys
import glob
import json
import math
import numpy as np
from pathlib import Path
from collections import namedtuple
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

# ================================================================
# CONSTANTS
# ================================================================
Planet = namedtuple("Planet", ["id", "owner", "x", "y", "radius", "ships", "production"])
CENTER_X, CENTER_Y = 50.0, 50.0
MAX_SPEED = 6.0
BOARD_SIZE = 100.0

# Danger zones from top-10 episodes
TRAPS = [
    (15, 83), (5, 67), (9, 76), (75, 75), (89, 61),
    (6, 28), (90, 68), (95, 69), (56, 9), (36, 18)
]

# ================================================================
# FEATURE ENGINEERING HELPERS
# ================================================================
def get_danger_heat(tx: float, ty: float) -> float:
    """Distance-based danger from traps [0, 1]"""
    if not TRAPS:
        return 0.0
    min_d = min(math.hypot(tx - t[0], ty - t[1]) for t in TRAPS)
    return max(0.0, min(1.0, 1.0 - (min_d / 15.0)))

def min_mirror_dist(tx: float, ty: float, sx: float, sy: float) -> float:
    """Distance to mirrored variants"""
    d1 = math.hypot(tx - (BOARD_SIZE - sx), ty - sy)
    d2 = math.hypot(tx - sx, ty - (BOARD_SIZE - sy))
    d3 = math.hypot(tx - (BOARD_SIZE - sx), ty - (BOARD_SIZE - sy))
    return min(d1, d2, d3)

# ================================================================
# PHYSICS CALCULATIONS
# ================================================================
def fleet_speed(ships: int) -> float:
    """Fleet speed based on ship count"""
    if ships <= 1:
        return 1.0
    log_factor = min(1.0, math.log(ships) / math.log(1000.0))
    return 1.0 + (MAX_SPEED - 1.0) * (log_factor ** 1.5)

def get_target_pos(
    tgt: Planet,
    turns: int,
    ang_vel: float,
    comets: List,
    comet_ids: set
) -> Tuple[float, float]:
    """Predict target position with orbital mechanics"""
    
    # Handle comets
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

def plan_flight(
    src: Planet,
    tgt: Planet,
    ships: int,
    ang_vel: float,
    comets: List,
    comet_ids: set
) -> Tuple[float, int, float, float, float]:
    """Plan intercept trajectory"""
    
    speed = fleet_speed(ships)
    tx, ty = tgt.x, tgt.y
    eta = 0.0
    
    for _ in range(5):
        dist = math.hypot(tx - src.x, ty - src.y)
        flight_dist = max(0.0, dist - src.radius - tgt.radius - 0.1)
        eta = flight_dist / speed if speed > 0 else 0.0
        tx, ty = get_target_pos(tgt, int(math.ceil(eta)), ang_vel, comets, comet_ids)
    
    angle = math.atan2(ty - src.y, tx - src.x)
    return angle, int(math.ceil(eta)), tx, ty, speed

# ================================================================
# NEURAL NETWORK (V6)
# ================================================================
class OrbitNetV6(nn.Module):
    """Improved network with batch norm, dropout, deeper architecture"""
    
    def __init__(self, input_dim: int = 18, dropout: float = 0.3):
        super().__init__()
        
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 1)
        
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(64)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        
        x = self.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        
        x = self.relu(self.bn4(self.fc4(x)))
        x = self.dropout(x)
        
        return torch.sigmoid(self.fc5(x))

# ================================================================
# LOSS FUNCTION (Focal Loss for Imbalanced Data)
# ================================================================
def focal_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 1.0,
    gamma: float = 2.0
) -> torch.Tensor:
    """Focal loss emphasizes hard negatives"""
    bce = torch.nn.functional.binary_cross_entropy(outputs, targets, reduction='none')
    p = torch.where(targets > 0.5, outputs, 1 - outputs)
    focal_weight = (1 - p) ** gamma
    return (alpha * focal_weight * bce).mean()

# ================================================================
# DATA EXTRACTION
# ================================================================
def compute_move_label(
    player_id: int,
    is_winner: bool,
    tgt: Planet,
    tx: float, ty: float,
    ships_sent: int,
    danger_heat: float
) -> float:
    """Compute nuanced label for move"""
    
    base = 0.7 if is_winner else 0.2
    
    # Bonus for capturing enemy territory
    if is_winner and tgt.owner not in [-1, player_id]:
        base += 0.2
    
    # Bonus for securing neutral
    if is_winner and tgt.owner == -1:
        base += 0.1
    
    # Penalty for dangerous zones
    if danger_heat > 0.6:
        base -= 0.3
    
    # Penalty for overcommitting
    if ships_sent >= 50:
        base -= 0.15
    
    return max(0.0, min(1.0, base))

def extract_training_chunk(
    files_chunk: List[str]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract features and labels from game files"""
    
    X_data, Y_data = [], []
    
    for filepath in files_chunk:
        try:
            with open(filepath, 'r') as f:
                match = json.load(f)
            
            steps = match.get("steps", [])
            if not steps:
                continue
            
            # Determine winner
            final_rewards = [a.get("reward", -1) for a in steps[-1]]
            if not final_rewards:
                continue
            
            try:
                winner_id = final_rewards.index(max(final_rewards))
            except ValueError:
                continue
            
            # Process each step
            for step_idx, step in enumerate(steps[:-25]):
                obs = step[0].get("observation", {})
                planets = [Planet(*p) for p in obs.get("planets", [])]
                ang_vel = obs.get("angular_velocity", 0.0)
                comets = obs.get("comets", [])
                comet_ids = set(obs.get("comet_planet_ids", []))
                
                # Compute player metrics
                player_ships = [sum(p.ships for p in planets if p.owner == pid) for pid in range(len(step))]
                player_territories = [sum(1 for p in planets if p.owner == pid) for pid in range(len(step))]
                
                for player_id in range(len(step)):
                    actions = step[player_id].get("action", [])
                    is_winner = (player_id == winner_id)
                    
                    for act in actions:
                        try:
                            src_id, angle, ships_sent = act[0], act[1], max(1, act[2])
                        except (IndexError, TypeError):
                            continue
                        
                        src_p = next((p for p in planets if p.id == src_id), None)
                        if not src_p:
                            continue
                        
                        # Find target by angle matching
                        best_diff, tgt_p = float('inf'), None
                        for p in planets:
                            if p.id == src_id:
                                continue
                            tgt_ang = math.atan2(p.y - src_p.y, p.x - src_p.x)
                            ang_diff = abs((angle - tgt_ang + math.pi) % (2 * math.pi) - math.pi)
                            if ang_diff < best_diff:
                                best_diff, tgt_p = ang_diff, p
                        
                        if best_diff > 0.2 or not tgt_p:
                            continue
                        
                        # Physics
                        try:
                            angle_res, eta, tx, ty, speed = plan_flight(
                                src_p, tgt_p, ships_sent, ang_vel, comets, comet_ids
                            )
                        except Exception:
                            continue
                        
                        if eta > 99:
                            continue
                        
                        # Extract 18 features
                        danger = get_danger_heat(tx, ty)
                        
                        feat = [
                            abs(tx - 50.0) / 50.0,                                           # 1. Folded X
                            abs(ty - 50.0) / 50.0,                                           # 2. Folded Y
                            tgt_p.production / 5.0,                                          # 3. Target production
                            min(1.0, tgt_p.ships / 100.0),                                   # 4. Target garrison
                            1.0 if ships_sent >= 20 else 0.0,                                # 5. Is significant
                            min(1.0, eta / 50.0),                                            # 6. Normalized ETA
                            danger,                                                          # 7. Danger heat
                            min(1.0, eta / max(1.0, src_p.ships)),                           # 8. Capital risk
                            min(1.0, tgt_p.production / max(1.0, float(ships_sent))),        # 9. ROI
                            tgt_p.production / 5.0,                                          # 10. Income gain
                            1.0 if tgt_p.owner == player_id else 0.0,                        # 11. Is defensive
                            1.0 if tgt_p.owner == -1 else (0.5 if tgt_p.owner == player_id else -0.5),  # 12. Owner
                            min(1.0, player_ships[player_id] / 500.0),                       # 13. Fleet size
                            min(1.0, player_territories[player_id] / 40.0),                  # 14. Territory
                            min(1.0, step_idx / 500.0),                                      # 15. Game phase
                            min(1.0, min([math.hypot(tx - p[0], ty - p[1]) / 50.0 for p in TRAPS] + [1.0])),  # 16. Trap
                            min(1.0, max(0.0, eta - 10) / 30.0),                             # 17. Flight duration
                            min(1.0, math.log(max(1, ships_sent) + 1) / math.log(1000)),     # 18. Log ships
                        ]
                        
                        # Compute label
                        label = compute_move_label(
                            player_id, is_winner, tgt_p, tx, ty, ships_sent, danger
                        )
                        
                        X_data.append(feat)
                        Y_data.append([label])
        
        except Exception as e:
            continue
    
    if not X_data:
        return torch.empty(0, 18), torch.empty(0, 1)
    
    return (
        torch.tensor(X_data, dtype=torch.float32),
        torch.tensor(Y_data, dtype=torch.float32)
    )

# ================================================================
# TRAINING LOOP
# ================================================================
def train_and_export(
    dataset_path: str = "/kaggle/input/datasets/bovard/orbit-wars-top10-episodes-2026-05-04/episodes/episodes",
    output_path: str = "/kaggle/working/orbit_model_v6.onnx"
) -> bool:
    """Train model and export to ONNX"""
    
    print("🚀 OrbitNet V6 Training Pipeline")
    print("=" * 60)
    
    # Find training files
    files = glob.glob(os.path.join(dataset_path, "**/*.json"), recursive=True)
    if not files:
        print(f"❌ No training files found at: {dataset_path}")
        return False
    
    print(f"📁 Found {len(files)} training files")
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")
    
    # Initialize model
    model = OrbitNetV6(input_dim=18, dropout=0.3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0003)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=3, verbose=False)
    
    chunk_size = 300
    best_loss = float('inf')
    total_samples = 0
    
    print(f"📊 Training on chunks of {chunk_size} files")
    print("=" * 60)
    
    # Process chunks
    for chunk_num in range(0, len(files), chunk_size):
        chunk_files = files[chunk_num:chunk_num + chunk_size]
        X_train, Y_train = extract_training_chunk(chunk_files)
        
        if len(X_train) == 0:
            print(f"⏭️  Chunk {chunk_num}: Skipped (no data)")
            continue
        
        print(f"\n📦 Chunk {chunk_num}: {len(X_train)} samples")
        total_samples += len(X_train)
        
        # Create dataloader
        dataset = TensorDataset(X_train.to(device), Y_train.to(device))
        dataloader = DataLoader(dataset, batch_size=512, shuffle=True, num_workers=0)
        
        # Train for multiple epochs
        for epoch in range(15):
            epoch_loss = 0.0
            
            for batch_X, batch_Y in dataloader:
                optimizer.zero_grad()
                
                outputs = model(batch_X)
                loss = focal_loss(outputs, batch_Y, gamma=2.0)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            
            if epoch % 3 == 0:
                print(f"   Epoch {epoch:2d}: Loss = {avg_loss:.6f}")
            
            scheduler.step(avg_loss)
            best_loss = min(best_loss, avg_loss)
    
    print("\n" + "=" * 60)
    print(f"✅ Training complete!")
    print(f"📈 Total samples processed: {total_samples}")
    print(f"📉 Best loss: {best_loss:.6f}")
    
    # Export to ONNX
    print(f"📤 Exporting to ONNX...")
    
    model.eval()
    dummy_input = torch.randn(1, 18).to(device)
    
    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            verbose=False
        )
        
        # Post-process ONNX
        import onnx
        onnx_model = onnx.load(output_path)
        onnx_model.ir_version = 9
        onnx.save(onnx_model, output_path)
        
        print(f"✅ Model saved to: {output_path}")
        return True
    
    except Exception as e:
        print(f"❌ Export failed: {e}")
        return False

# ================================================================
# ENTRY POINT
# ================================================================
if __name__ == "__main__":
    success = train_and_export()
    sys.exit(0 if success else 1)
