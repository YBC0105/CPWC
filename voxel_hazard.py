import numpy as np
import pandas as pd
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set
import math

Vec3 = Tuple[float, float, float]

@dataclass
class VoxelConfig:
    # Unity 맵 범위에 맞춘 공간 설정 (10000m × 10000m)
    world_min: Vec3 = (0.0, 8.0, 0.0)        # Unity 맵 시작점 (높이 8-12m 범위)
    world_max: Vec3 = (10000.0, 12.0, 10000.0)  # Unity 맵 끝점
    voxel_size: float = 0.5                   # 복셀 크기 0.5m³ (showvoxel.py와 동일)
    
    # 실제 Unity 차량 크기 반영 + 현실적 형태
    ego_size: Vec3 = (4.0, 2.0, 6.0)         # 차량: Unity 실측 크기 (3단계 모델)
    actor_size: Vec3 = (1.2, 1.8, 0.8)       # 사람: 실제 체형 (4단계 모델)
    
    # 예측 설정
    time_horizon_s: float = 3.0               # 3초 후 예측
    
    # 확률 계산 설정
    voxel_sphere_radius: float = 1.5          # 각 복셀 중심의 구체 반지름 (미터)
    overlap_detection_distance: float = 15.0  # 15m 이내에서 근접 감지
    max_probability: float = 100.0            # 최대 확률

@dataclass
class VoxelEvent:
    actor_id: str
    t_sec: float
    ego_center: Vec3
    actor_center: Vec3
    ego_voxels: Set[Tuple[int, int, int]]     # 차량이 점유한 복셀들
    actor_voxels: Set[Tuple[int, int, int]]   # 사람이 점유한 복셀들
    overlap_sphere_pairs: int                 # 접촉하는 구체 쌍의 개수
    total_ego_voxels: int                     # 차량 전체 복셀 개수
    total_actor_voxels: int                   # 사람 전체 복셀 개수
    overlap_probability: float                # 겹침 확률 (0-100%)
    overlapping_actor_count: float            # 접촉하는 사람 복셀 개수
    distance_m: float                         # 중심점 간 거리

class VoxelHazardDetector:
    """
    3D Voxel 기반 위험 감지 시스템 (중심점 정렬 수정됨)
    - 3D 공간을 1m³ 복셀로 분할
    - 차량과 사람의 3초 후 예측 위치를 복셀로 변환
    - 겹치는 복셀 개수로 위험도 계산
    - 거리 기반 확률 가중치 적용
    """
    
    def __init__(self, config: VoxelConfig):
        self.cfg = config
        
        # 복셀 그리드 크기 계산
        self.grid_size = (
            int((config.world_max[0] - config.world_min[0]) / config.voxel_size),
            int((config.world_max[1] - config.world_min[1]) / config.voxel_size),
            int((config.world_max[2] - config.world_min[2]) / config.voxel_size)
        )
        
        print(f"🧊 Voxel 그리드 초기화: {self.grid_size[0]}×{self.grid_size[1]}×{self.grid_size[2]} = {np.prod(self.grid_size):,}개 복셀")
        print(f"🗺️ Unity 맵 범위: {config.world_min} ~ {config.world_max}")
        print(f"🚗 차량 크기: {config.ego_size[0]:.1f}×{config.ego_size[1]:.1f}×{config.ego_size[2]:.1f}m (3단계 현실 모델)")
        print(f"🚶 사람 크기: {config.actor_size[0]:.1f}×{config.actor_size[1]:.1f}×{config.actor_size[2]:.1f}m (4단계 인체 모델)")
        print(f"📏 복셀 크기: {config.voxel_size}m³ (showvoxel.py와 동일)")
        print(f"🔮 복셀 구체 반지름: {config.voxel_sphere_radius}m (겹침 감지용)")
        print(f"🎯 혁신 기술: showvoxel.py 방식 적용 - 정확한 중심점 정렬")
        print(f"✅ 좌표 변환: center 파라미터 + 0.5 오프셋 방식")
        print(f"📈 복셀 개수 증가: 0.5m³ 복셀로 더 정밀한 형태 표현")
    
    def world_to_voxel(self, pos: Vec3, center: Vec3 = (0, 0, 0)) -> Tuple[int, int, int]:
        """월드 좌표를 복셀 인덱스로 변환 (showvoxel.py 방식)"""
        x = int((pos[0] - center[0]) / self.cfg.voxel_size)
        y = int((pos[1] - center[1]) / self.cfg.voxel_size)
        z = int((pos[2] - center[2]) / self.cfg.voxel_size)
        return (x, y, z)
    
    def voxel_to_world(self, voxel: Tuple[int, int, int], center: Vec3 = (0, 0, 0)) -> Vec3:
        """복셀 인덱스를 월드 좌표로 변환 (showvoxel.py 방식)"""
        x = center[0] + (voxel[0] + 0.5) * self.cfg.voxel_size
        y = center[1] + (voxel[1] + 0.5) * self.cfg.voxel_size
        z = center[2] + (voxel[2] + 0.5) * self.cfg.voxel_size
        return (x, y, z)
    
    def get_car_realistic_voxels(self, center: Vec3, size: Vec3) -> Set[Tuple[int, int, int]]:
        """
        실제 차량 형태를 반영한 복셀 생성 (showvoxel.py 방식 적용)
        - 하단: 바퀴 부분 (양쪽 끝만)
        - 중단: 차체 본체 (직육면체)
        - 상단: 루프 부분 (중앙 부분만, 앞뒤는 짧게)
        """
        voxels = set()
        half = (size[0]/2, size[1]/2, size[2]/2)
        min_pos = (center[0]-half[0], center[1]-half[1], center[2]-half[2])
        max_pos = (center[0]+half[0], center[1]+half[1], center[2]+half[2])
        min_v = self.world_to_voxel(min_pos)
        max_v = self.world_to_voxel(max_pos)

        y_range = max_v[1] - min_v[1] + 1
        middle_y = min_v[1] + y_range//3
        top_y = min_v[1] + (y_range*2)//3

        for x in range(min_v[0], max_v[0]+1):
            for y in range(min_v[1], max_v[1]+1):
                for z in range(min_v[2], max_v[2]+1):
                    # Unity 맵 범위 체크 (선택적)
                    world_pos = self.voxel_to_world((x, y, z))
                    if not (self.cfg.world_min[0] <= world_pos[0] <= self.cfg.world_max[0] and
                            self.cfg.world_min[1] <= world_pos[1] <= self.cfg.world_max[1] and
                            self.cfg.world_min[2] <= world_pos[2] <= self.cfg.world_max[2]):
                        continue
                    
                    if y < middle_y:
                        wheel_front = abs(z - min_v[2]) <= 1
                        wheel_rear  = abs(z - max_v[2]) <= 1
                        wheel_side  = (x == min_v[0] or x == max_v[0])
                        if (wheel_front or wheel_rear) and wheel_side:
                            voxels.add((x,y,z))
                    elif y < top_y:
                        voxels.add((x,y,z))
                    else:
                        x_margin = (max_v[0]-min_v[0])//4
                        z_margin = (max_v[2]-min_v[2])//3
                        if (min_v[0]+x_margin <= x <= max_v[0]-x_margin and
                            min_v[2]+z_margin <= z <= max_v[2]-z_margin):
                            voxels.add((x,y,z))
        return voxels

    def get_person_realistic_voxels(self, center: Vec3, size: Vec3) -> Set[Tuple[int, int, int]]:
        """
        실제 사람 형태를 반영한 복셀 생성 (showvoxel.py 방식 적용)
        - 하단: 다리 부분 (좌우 분리된 두 기둥)
        - 중단: 몸통 부분 (어깨 너비, 허리 좁음)
        - 상단: 머리 부분 (작은 원형)
        """
        voxels = set()
        half = (size[0]/2, size[1]/2, size[2]/2)
        min_pos = (center[0]-half[0], center[1]-half[1], center[2]-half[2])
        max_pos = (center[0]+half[0], center[1]+half[1], center[2]+half[2])
        min_v = self.world_to_voxel(min_pos)
        max_v = self.world_to_voxel(max_pos)

        y_range = max_v[1] - min_v[1] + 1
        legs_y_end  = min_v[1] + y_range//3
        torso_y_end = min_v[1] + (y_range*3)//4

        cx = (min_v[0] + max_v[0]) / 2
        cz = (min_v[2] + max_v[2]) / 2

        for x in range(min_v[0], max_v[0]+1):
            for y in range(min_v[1], max_v[1]+1):
                for z in range(min_v[2], max_v[2]+1):
                    # Unity 맵 범위 체크 (선택적)
                    world_pos = self.voxel_to_world((x, y, z))
                    if not (self.cfg.world_min[0] <= world_pos[0] <= self.cfg.world_max[0] and
                            self.cfg.world_min[1] <= world_pos[1] <= self.cfg.world_max[1] and
                            self.cfg.world_min[2] <= world_pos[2] <= self.cfg.world_max[2]):
                        continue
                    
                    if y <= legs_y_end:
                        leg_w = max(1, (max_v[0]-min_v[0])//3)
                        left  = (min_v[0] <= x <= min_v[0]+leg_w)
                        right = (max_v[0]-leg_w <= x <= max_v[0])
                        if left or right:
                            voxels.add((x,y,z))
                    elif y <= torso_y_end:
                        shoulder = (max_v[0]-min_v[0])
                        waist = shoulder * 0.7
                        prog = (y - legs_y_end) / max(1, (torso_y_end - legs_y_end))
                        cur_w = shoulder - (shoulder - waist) * prog
                        margin = (max_v[0]-min_v[0]-cur_w)/2
                        if (min_v[0]+margin <= x <= max_v[0]-margin):
                            voxels.add((x,y,z))
                    else:
                        rx = (max_v[0]-min_v[0]) * 0.3
                        rz = (max_v[2]-min_v[2]) * 0.3
                        dx = (x - cx) / max(rx, 0.5)
                        dz = (z - cz) / max(rz, 0.5)
                        if math.sqrt(dx*dx + dz*dz) <= 1.0:
                            voxels.add((x,y,z))
        return voxels
    
    def calculate_sphere_overlap_probability(self, ego_voxels: Set, actor_voxels: Set, voxel_radius: float = 1.0) -> Tuple[float, int, float]:
        """
        각 복셀을 중심으로 하는 구체들의 접촉 개수로 확률 계산
        
        확률 매핑 규칙:
        - 0개 접촉: 0% (안전)
        - 1개 이상 접촉: 50% (접촉 시작)
        - 사람 절반 접촉: 100% (최대 위험)
        - 중간: 선형 증가
        
        Parameters:
        -----------
        ego_voxels : Set
            차량 복셀들
        actor_voxels : Set  
            사람 복셀들
        voxel_radius : float
            각 복셀 중심의 구체 반지름 (미터)
            
        Returns:
        --------
        Tuple[float, int, float]
            (위험확률%, 접촉하는 구체 쌍 개수, 접촉하는 사람 복셀 개수)
        """
        if len(ego_voxels) == 0 or len(actor_voxels) == 0:
            return 0.0, 0, 0.0
        
        overlap_pairs = 0
        overlapping_actor_voxels = set()  # 접촉하는 사람 복셀들
        
        # 각 차량 복셀과 사람 복셀 간의 거리 체크
        for ego_voxel in ego_voxels:
            ego_world_pos = self.voxel_to_world(ego_voxel)
            
            for actor_voxel in actor_voxels:
                actor_world_pos = self.voxel_to_world(actor_voxel)
                
                # 두 복셀 중심 간의 거리 계산
                distance = math.sqrt(
                    (ego_world_pos[0] - actor_world_pos[0])**2 +
                    (ego_world_pos[1] - actor_world_pos[1])**2 +
                    (ego_world_pos[2] - actor_world_pos[2])**2
                )
                
                # 두 구체가 접촉하는지 확인 (거리 < 반지름1 + 반지름2)
                if distance < (voxel_radius * 2):  # 각각 같은 반지름
                    overlap_pairs += 1
                    overlapping_actor_voxels.add(actor_voxel)
        
        # 접촉하는 사람 복셀 개수
        overlapping_actor_count = len(overlapping_actor_voxels)
        
        if overlapping_actor_count == 0:
            return 0.0, overlap_pairs, 0.0
        
        # 위험 확률 계산 (요구사항에 따라)
        total_actor_voxels = len(actor_voxels)
        half_actor_voxels = total_actor_voxels / 2.0
        
        if overlapping_actor_count >= half_actor_voxels:
            # 사람 절반 이상 접촉: 100% (최대 위험)
            probability = 100.0
        else:
            # 1개 이상 접촉 ~ 사람 절반: 50% ~ 100% 선형 증가
            # overlapping_actor_count / half_actor_voxels 비율로 50%에서 100%까지
            ratio = overlapping_actor_count / half_actor_voxels  # 0.0 ~ 1.0
            probability = 50.0 + (ratio * 50.0)  # 50% ~ 100%
        
        return min(100.0, probability), overlap_pairs, float(overlapping_actor_count)
    
    @staticmethod
    def _dist_3d(a: Vec3, b: Vec3) -> float:
        """3D 거리 계산"""
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2 + (a[2] - b[2])**2)
    
    def detect(self, ego_position: Vec3, actors: Dict[str, Vec3]) -> List[VoxelEvent]:
        """
        Voxel 기반 위험 감지 (중심점 정렬 개선됨)
        
        Parameters:
        -----------
        ego_position : Vec3
            자차 3초 후 예측 위치
        actors : Dict[str, Vec3]
            액터들의 3초 후 예측 위치 {actor_id: position}
            
        Returns:
        --------
        List[VoxelEvent]
            각 액터별 위험 감지 결과
        """
        results = []
        
        # 자차 복셀 계산 (현실적 차량 형태, 중심점 정렬됨)
        ego_voxels = self.get_car_realistic_voxels(ego_position, self.cfg.ego_size)
        
        for actor_id, actor_position in actors.items():
            # 사람 복셀 계산 (현실적 인체 형태, 중심점 정렬됨)
            actor_voxels = self.get_person_realistic_voxels(actor_position, self.cfg.actor_size)
            
            # 구체 접촉 기반 확률 계산 (접촉 개수 방식)
            probability, overlap_pairs, overlapping_actor_count = self.calculate_sphere_overlap_probability(
                ego_voxels, actor_voxels, self.cfg.voxel_sphere_radius
            )
            
            # 거리 계산
            distance = self._dist_3d(ego_position, actor_position)
            
            # 위험 여부 (구체 겹침이 있거나 일정 거리 이내)
            is_dangerous = (overlap_pairs > 0 or 
                          distance <= self.cfg.overlap_detection_distance)
            
            result = VoxelEvent(
                actor_id=actor_id,
                t_sec=self.cfg.time_horizon_s,
                ego_center=ego_position,
                actor_center=actor_position,
                ego_voxels=ego_voxels,
                actor_voxels=actor_voxels,
                overlap_sphere_pairs=overlap_pairs,
                total_ego_voxels=len(ego_voxels),
                total_actor_voxels=len(actor_voxels),
                overlap_probability=probability,
                overlapping_actor_count=overlapping_actor_count,
                distance_m=distance
            )
            
            results.append(result)
        
        return results

class VoxelHazardMonitor:
    """
    Voxel 기반 실시간 위험 모니터링 시스템 (중심점 정렬 개선됨)
    """
    
    def __init__(self, 
                 ego_id: str = "1003",
                 actor_ids: List[str] = ["1001"],
                 config: Optional[VoxelConfig] = None):
        
        self.ego_id = ego_id
        self.actor_ids = actor_ids
        self.cfg = config or VoxelConfig()
        self.detector = VoxelHazardDetector(self.cfg)
        self.last_processed_second: Optional[int] = None
        
        print("🎯 Voxel 기반 위험 모니터링 시스템 시작 (showvoxel.py 방식 적용)")
        print(f"📍 자차 ID: {ego_id}, 감시 대상: {actor_ids}")
        print("=" * 60)
    
    def _linear_predict_xyz(self, hist: pd.DataFrame, horizon_s: float) -> Optional[Vec3]:
        """선형 예측으로 미래 위치 계산"""
        if hist is None or len(hist) < 2:
            return None
        
        t0 = float(hist['time'].iloc[0])
        t1 = float(hist['time'].iloc[-1])
        if t1 <= t0:
            return None
        
        dt = t1 - t0
        x_speed = (float(hist['x'].iloc[-1]) - float(hist['x'].iloc[0])) / dt
        y_speed = (float(hist['y'].iloc[-1]) - float(hist['y'].iloc[0])) / dt
        z_speed = (float(hist['z'].iloc[-1]) - float(hist['z'].iloc[0])) / dt
        
        last = hist.iloc[-1]
        pred_x = float(last['x']) + x_speed * horizon_s
        pred_y = float(last['y']) + y_speed * horizon_s
        pred_z = float(last['z']) + z_speed * horizon_s
        
        return (pred_x, pred_y, pred_z)
    
    def _load_latest_dataframe(self) -> Optional[pd.DataFrame]:
        """최신 데이터 로드"""
        data_file = "realtime_data.csv"
        if not os.path.exists(data_file):
            return None
        try:
            df = pd.read_csv(data_file)
            return df if len(df) > 0 else None
        except Exception as e:
            print(f"[Voxel Monitor] CSV 읽기 오류: {e}")
            return None
    
    def run(self):
        """메인 모니터링 루프"""
        print("🔍 showvoxel.py 방식 적용된 Voxel 모니터링 시작 (Ctrl+C로 종료)")
        
        while True:
            try:
                df = self._load_latest_dataframe()
                if df is None:
                    time.sleep(0.2)
                    continue
                
                # 초 단위 처리
                cur_sec = int(df['time'].max())
                if self.last_processed_second == cur_sec:
                    time.sleep(0.05)
                    continue
                self.last_processed_second = cur_sec
                
                # 데이터 필터링
                df['custom_id'] = df['custom_id'].astype(str)
                ego_df = df[df['custom_id'] == self.ego_id].copy()
                actor_df = df[df['custom_id'].isin(self.actor_ids)].copy()
                
                if len(ego_df) < 2:
                    print(f"[{cur_sec}s] 자차 데이터 부족")
                    time.sleep(0.05)
                    continue
                
                # 자차 예측
                ego_pred = self._linear_predict_xyz(ego_df.sort_values('time'), self.cfg.time_horizon_s)
                if ego_pred is None:
                    print(f"[{cur_sec}s] 자차 예측 실패")
                    time.sleep(0.05)
                    continue
                
                # 액터 예측
                actors_pred = {}
                for aid in self.actor_ids:
                    hist = actor_df[actor_df['custom_id'] == aid].sort_values('time').tail(150)
                    if len(hist) < 2:
                        continue
                    a_pred = self._linear_predict_xyz(hist, self.cfg.time_horizon_s)
                    if a_pred is not None:
                        actors_pred[str(aid)] = a_pred
                
                if not actors_pred:
                    print(f"[{cur_sec}s] 액터 데이터 부족")
                    time.sleep(0.05)
                    continue
                
                # 중심점 정렬된 Voxel 기반 위험 감지
                events = self.detector.detect(ego_pred, actors_pred)
                
                # 복셀 중심점 검증 (주기적으로) - showvoxel.py 방식으로 개선
                if cur_sec % 10 == 0:  # 10초마다
                    for e in events:
                        if len(e.ego_voxels) > 0:
                            # 월드 좌표로 변환하여 실제 중심 계산
                            ego_world_coords = np.array([self.detector.voxel_to_world(v) for v in e.ego_voxels])
                            ego_center_actual = np.mean(ego_world_coords, axis=0)
                            deviation_ego = np.sqrt(sum((ego_center_actual[i] - e.ego_center[i])**2 for i in range(3)))
                            print(f"[검증] 차량 중심 - 입력: {tuple(round(v,1) for v in e.ego_center)}, 실제: ({ego_center_actual[0]:.1f},{ego_center_actual[1]:.1f},{ego_center_actual[2]:.1f}), 편차: {deviation_ego:.2f}m")
                        
                        if len(e.actor_voxels) > 0:
                            # 월드 좌표로 변환하여 실제 중심 계산
                            actor_world_coords = np.array([self.detector.voxel_to_world(v) for v in e.actor_voxels])
                            actor_center_actual = np.mean(actor_world_coords, axis=0)
                            deviation_actor = np.sqrt(sum((actor_center_actual[i] - e.actor_center[i])**2 for i in range(3)))
                            print(f"[검증] 사람 중심 - 입력: {tuple(round(v,1) for v in e.actor_center)}, 실제: ({actor_center_actual[0]:.1f},{actor_center_actual[1]:.1f},{actor_center_actual[2]:.1f}), 편차: {deviation_actor:.2f}m")
                
                # 결과 출력 (접촉 개수 기반 확률 표기)
                any_danger = False
                for e in events:
                    if e.overlap_sphere_pairs > 0:
                        any_danger = True
                        total_possible = e.total_ego_voxels * e.total_actor_voxels
                        contact_percentage = (e.overlap_sphere_pairs / total_possible * 100) if total_possible > 0 else 0
                        half_person_voxels = e.total_actor_voxels / 2.0
                        
                        print(f"🚨[위험-구체접촉] t+{e.t_sec:.0f}s {e.actor_id}")
                        print(f"   접촉쌍: {e.overlap_sphere_pairs}쌍/{total_possible}쌍 (접촉률:{contact_percentage:.1f}%)")
                        print(f"   🎯 위험확률: {e.overlap_probability:.1f}%")
                        print(f"   📊 사람복셀접촉: {int(e.overlapping_actor_count)}개/{e.total_actor_voxels}개 (절반:{half_person_voxels:.1f}개)")
                        print(f"   📏 거리: {e.distance_m:.2f}m")
                        print(f"   📍 위치 - 차:{tuple(round(v,1) for v in e.ego_center)}, 사람:{tuple(round(v,1) for v in e.actor_center)}")
                        
                        # 접촉 개수별 설명
                        if e.overlapping_actor_count >= half_person_voxels:
                            print(f"   🔴 위험도: 최대 (사람 절반 이상 접촉)")
                        elif e.overlapping_actor_count >= 1:
                            print(f"   🟡 위험도: 진행중 (접촉 시작~절반)")
                        
                    elif e.distance_m <= self.cfg.overlap_detection_distance:
                        print(f"👀[근접감지] t+{e.t_sec:.0f}s {e.actor_id} 거리: {e.distance_m:.2f}m (접촉없음)")
                        print(f"   🎯 위험확률: {e.overlap_probability:.1f}%")
                        print(f"   📊 사람복셀접촉: {int(e.overlapping_actor_count)}개/{e.total_actor_voxels}개")
                
                if not any_danger:
                    nearest = min(events, key=lambda ev: ev.distance_m)
                    print(f"✅[안전] t+{nearest.t_sec:.0f}s 최근접 {nearest.actor_id} "
                          f"거리:{nearest.distance_m:.2f}m")
                    print(f"   🎯 위험확률: {nearest.overlap_probability:.1f}%")
                    print(f"   📊 사람복셀접촉: {int(nearest.overlapping_actor_count)}개/{nearest.total_actor_voxels}개")
                
                # 위험 확률 계산 규칙 설명 (30초마다)
                if cur_sec % 30 == 0:  # 30초마다
                    print(f"\n💡[위험확률 규칙] 접촉 개수 기반:")
                    print(f"   - 0개 접촉: 0% (안전)")
                    print(f"   - 1개 이상 접촉: 50% (접촉 시작)")
                    print(f"   - 사람 절반 접촉: 100% (최대 위험)")
                    print(f"   - 중간: 선형 증가")
                    print(f"   - 복셀 크기: {self.cfg.voxel_size}m³ (더 정밀한 감지)")
                
                print("─" * 50)
                
                print("─" * 50)
                
                time.sleep(0.05)
                
            except KeyboardInterrupt:
                print("\n🛑 showvoxel.py 방식 적용된 Voxel 모니터 종료")
                break
            except Exception as e:
                print(f"[Voxel Monitor] 오류: {e}")
                time.sleep(0.2)

def main():
    """메인 실행 함수"""
    config = VoxelConfig(
        world_min=(0.0, 8.0, 0.0),          # Unity 맵 시작 (높이 8-12m)
        world_max=(10000.0, 12.0, 10000.0), # Unity 맵 전체 (10km×4m×10km)
        voxel_size=0.5,                     # 0.5m³ 복셀 (showvoxel.py와 동일)
        ego_size=(4.0, 2.0, 6.0),           # 차량: 현실적 3단계 모델
        actor_size=(1.2, 1.8, 0.8),         # 사람: 현실적 4단계 모델
        time_horizon_s=3.0,
        voxel_sphere_radius=1.5,            # 각 복셀 중심의 구체 반지름 1.5m
        overlap_detection_distance=15.0     # 15m 이내에서 근접 감지
    )
    
    monitor = VoxelHazardMonitor(
        ego_id="1003",
        actor_ids=["1001"],
        config=config
    )
    
    monitor.run()

if __name__ == "__main__":
    main()