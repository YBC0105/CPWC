import numpy as np
import pandas as pd
import os
import time
import math
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Set

Vec3 = Tuple[float, float, float]

def _normalize2(vx: float, vz: float) -> Tuple[float, float]:
    n = math.hypot(vx, vz)
    if n <= 1e-9:
        return (1.0, 0.0)
    return (vx / n, vz / n)

class VoxelHazardDetector:
    def __init__(self, voxel_size: float = 0.22, sphere_radius: float = 1.0):
        self.voxel_size = voxel_size
        self.sphere_radius = sphere_radius
        self.car_size = (3.8, 2.0, 5.8)
        self.person_size = (0.6, 1.7, 0.6)

    @staticmethod
    def yaw_from_forward(forward: Vec3) -> float:
        fx, _, fz = forward
        return math.atan2(fz, fx)

    @staticmethod
    def rotate_yaw(p: Vec3, yaw: float) -> Vec3:
        x, y, z = p
        c, s = math.cos(yaw), math.sin(yaw)
        xr = x * c - z * s
        zr = x * s + z * c
        return (xr, y, zr)

    def world_to_local(self, p_world: Vec3, center: Vec3, yaw: float) -> Vec3:
        px = p_world[0] - center[0]
        py = p_world[1] - center[1]
        pz = p_world[2] - center[2]
        c, s = math.cos(-yaw), math.sin(-yaw)
        x = px * c - pz * s
        z = px * s + pz * c
        return (x, py, z)

    def create_voxels(self, center: Vec3, size: Vec3, is_car: bool = False) -> Set[Tuple[int, int, int]]:
        voxels = set()
        half = (size[0]/2, size[1]/2, size[2]/2)

        def idx_min(val: float) -> int:
            return int(math.floor(val / self.voxel_size))

        def idx_max(val: float) -> int:
            return int(math.ceil(val / self.voxel_size) - 1)

        min_v = (idx_min(center[0] - half[0]),
                 idx_min(center[1] - half[1]),
                 idx_min(center[2] - half[2]))
        max_v = (idx_max(center[0] + half[0]),
                 idx_max(center[1] + half[1]),
                 idx_max(center[2] + half[2]))

        for x in range(min_v[0], max_v[0] + 1):
            for y in range(min_v[1], max_v[1] + 1):
                for z in range(min_v[2], max_v[2] + 1):
                    voxels.add((x, y, z))

        return voxels

    def _obb_aabb_bounds(self, center: Vec3, size: Vec3, yaw: float) -> Tuple[Vec3, Vec3]:
        L, H, W = size
        hx, hy, hz = L/2, H/2, W/2
        corners_local = [(sx*hx, sy*hy, sz*hz)
                         for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
        corners_world = []
        for cx, cy, cz in corners_local:
            wx, wy, wz = self.rotate_yaw((cx, cy, cz), yaw)
            corners_world.append((wx + center[0], wy + center[1], wz + center[2]))
        xs = [p[0] for p in corners_world]
        ys = [p[1] for p in corners_world]
        zs = [p[2] for p in corners_world]
        min_w = (min(xs), min(ys), min(zs))
        max_w = (max(xs), max(ys), max(zs))
        return min_w, max_w

    def create_voxels_oriented(self, center: Vec3, size: Vec3, forward: Vec3) -> Set[Tuple[int, int, int]]:
        fx, _, fz = forward
        fx, fz = _normalize2(fx, fz)
        yaw = math.atan2(fz, fx)
        min_w, max_w = self._obb_aabb_bounds(center, size, yaw)

        def idx_min(val: float) -> int:
            return int(math.floor(val / self.voxel_size))

        def idx_max(val: float) -> int:
            return int(math.ceil(val / self.voxel_size) - 1)

        min_idx = (idx_min(min_w[0]), idx_min(min_w[1]), idx_min(min_w[2]))
        max_idx = (idx_max(max_w[0]), idx_max(max_w[1]), idx_max(max_w[2]))

        voxels = set()
        L, H, W = size
        hx, hy, hz = L/2.0, H/2.0, W/2.0
        eps = 1e-9

        for ix in range(min_idx[0], max_idx[0] + 1):
            cx = ix * self.voxel_size + self.voxel_size * 0.5
            for iy in range(min_idx[1], max_idx[1] + 1):
                cy = iy * self.voxel_size + self.voxel_size * 0.5
                for iz in range(min_idx[2], max_idx[2] + 1):
                    cz = iz * self.voxel_size + self.voxel_size * 0.5
                    lx, ly, lz = self.world_to_local((cx, cy, cz), center, yaw)
                    if (abs(lx) <= hx + eps) and (abs(ly) <= hy + eps) and (abs(lz) <= hz + eps):
                        voxels.add((ix, iy, iz))

        return voxels

    def voxel_to_world(self, voxel: Tuple[int, int, int]) -> Vec3:
        return (voxel[0] * self.voxel_size + self.voxel_size/2,
                voxel[1] * self.voxel_size + self.voxel_size/2,
                voxel[2] * self.voxel_size + self.voxel_size/2)

    def calculate_overlap(self, car_voxels: Set[Tuple[int,int,int]], person_voxels: Set[Tuple[int,int,int]]) -> Tuple[float, int, int]:
        if not car_voxels or not person_voxels:
            return 0.0, 0, 0
        overlap_voxels = car_voxels & person_voxels
        overlap_count = len(overlap_voxels)
        ratio = overlap_count / max(len(person_voxels), 1)
        if ratio >= 0.8:
            probability = 99.9
        elif ratio >= 0.3:
            probability = 50.0 + ((ratio - 0.3) / 0.5) * 49.9
        else:
            probability = ratio * 50.0 / 0.3
        return min(99.9, probability), overlap_count, overlap_count

    def detect_danger(self, car_pos: Vec3, person_pos: Vec3, actor_id: str, car_forward: Optional[Vec3] = None) -> Dict:
        if car_forward is not None:
            car_voxels = self.create_voxels_oriented(car_pos, self.car_size, car_forward)
        else:
            car_voxels = self.create_voxels(car_pos, self.car_size, is_car=True)
        person_voxels = self.create_voxels(person_pos, self.person_size, is_car=False)
        probability, overlap_pairs, overlap_count = self.calculate_overlap(car_voxels, person_voxels)
        distance = math.sqrt(sum((car_pos[i] - person_pos[i])**2 for i in range(3)))
        is_danger = (probability >= 50.0) or (distance <= 1.5)
        return {
            'actor_id': actor_id,
            'car_voxels': len(car_voxels),
            'person_voxels': len(person_voxels),
            'overlap_pairs': overlap_pairs,
            'overlap_count': overlap_count,
            'probability': probability,
            'distance': distance,
            'is_danger': is_danger
        }

class VoxelMonitor:
    def __init__(self, car_id: str = "1003", person_id: str = "1001"):
        self.car_id = car_id
        self.person_id = person_id
        self.detector = VoxelHazardDetector()
        self.last_second = None
        self.last_car_dir: Vec3 = (1.0, 0.0, 0.0)

    def save_to_json(self, result: Dict, car_pos: Vec3, person_pos: Vec3, time_sec: int, car_forward: Vec3):
        overlap_ratio = round(result['overlap_count'] / max(result['person_voxels'], 1) * 100, 1)
        data = {
            "timestamp": datetime.now().isoformat(),
            "time_second": time_sec,
            "prediction_time": time_sec + 3,
            "collision_probability": round(result['probability'], 1),
            "distance": round(result['distance'], 2),
            "is_danger": result['is_danger'],
            "status": "DANGER" if result['is_danger'] else "SAFE",
            "car_voxels": result['car_voxels'],
            "person_voxels": result['person_voxels'],
            "overlap_pairs": result['overlap_pairs'],
            "overlap_count": result['overlap_count'],
            "overlap_ratio": overlap_ratio,
            "car_position": {"x": round(car_pos[0], 2), "y": round(car_pos[1], 2), "z": round(car_pos[2], 2)},
            "person_position": {"x": round(person_pos[0], 2), "y": round(person_pos[1], 2), "z": round(person_pos[2], 2)},
            "car_heading": {"x": round(car_forward[0], 3), "y": round(car_forward[1], 3), "z": round(car_forward[2], 3)}
        }
        try:
            with open("hazard_status.json", "w", encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            if result['is_danger']:
                with open("danger_alert.json", "w", encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                print(f"🚨 DANGER: {result['probability']:.1f}%, {result['distance']:.1f}m")
        except Exception as e:
            print(f"JSON save error: {e}")

    def check_danger_alerts(self):
        pass

    def predict_position(self, data: pd.DataFrame, horizon: float = 3.0) -> Optional[Vec3]:
        if len(data) < 2:
            return None
        try:
            data = data.sort_values('time')
            dt = float(data['time'].iloc[-1] - data['time'].iloc[0])
            if dt <= 0:
                return None
            dx = (float(data['x'].iloc[-1]) - float(data['x'].iloc[0])) / dt
            dy = (float(data['y'].iloc[-1]) - float(data['y'].iloc[0])) / dt
            dz = (float(data['z'].iloc[-1]) - float(data['z'].iloc[0])) / dt
            last = data.iloc[-1]
            return (float(last['x']) + dx * horizon,
                    float(last['y']) + dy * horizon,
                    float(last['z']) + dz * horizon)
        except:
            return None

    def estimate_heading_from_logs(self, data: pd.DataFrame, K: int = 5, v_min: float = 0.1, alpha: float = 0.8) -> Vec3:
        try:
            if len(data) < 2:
                return self.last_car_dir
            df = data.sort_values('time')
            k = min(K, len(df)-1)
            p0 = df.iloc[-1 - k]
            p1 = df.iloc[-1]
            dt = float(p1['time'] - p0['time'])
            if dt <= 1e-6:
                return self.last_car_dir
            vx = (float(p1['x']) - float(p0['x'])) / dt
            vz = (float(p1['z']) - float(p0['z'])) / dt
            speed = math.hypot(vx, vz)
            if speed < v_min:
                return self.last_car_dir
            dx, dz = _normalize2(vx, vz)
            prev = self.last_car_dir
            mixed = (alpha * prev[0] + (1 - alpha) * dx, 0.0, alpha * prev[2] + (1 - alpha) * dz)
            nx, nz = _normalize2(mixed[0], mixed[2])
            self.last_car_dir = (nx, 0.0, nz)
            return self.last_car_dir
        except Exception:
            return self.last_car_dir

    def load_data(self) -> Optional[pd.DataFrame]:
        try:
            if not os.path.exists("realtime_data.csv"):
                return None
            df = pd.read_csv("realtime_data.csv")
            return df if len(df) > 0 else None
        except:
            return None

    def run(self):
        while True:
            try:
                self.check_danger_alerts()
                df = self.load_data()
                if df is None:
                    time.sleep(0.2)
                    continue
                cur_sec = int(df['time'].max())
                if self.last_second == cur_sec:
                    time.sleep(0.1)
                    continue
                self.last_second = cur_sec
                df['custom_id'] = df['custom_id'].astype(str)
                car_data = df[df['custom_id'] == self.car_id].tail(20)
                person_data = df[df['custom_id'] == self.person_id].tail(20)
                if len(car_data) < 2 or len(person_data) < 2:
                    continue
                car_pred = self.predict_position(car_data)
                person_pred = self.predict_position(person_data)
                if car_pred is None or person_pred is None:
                    continue
                car_forward = self.estimate_heading_from_logs(car_data)
                result = self.detector.detect_danger(car_pred, person_pred, self.person_id, car_forward=car_forward)
                self.save_to_json(result, car_pred, person_pred, cur_sec, car_forward)
                time.sleep(0.1)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(0.2)

def main():
    monitor = VoxelMonitor()
    monitor.run()

if __name__ == "__main__":
    main()
