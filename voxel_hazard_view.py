import numpy as np
import pandas as pd
import os
import time
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Set

Vec3 = Tuple[float, float, float]

class VoxelHazardDetector:
    """경량화된 Voxel 기반 위험 감지 시스템"""
    
    def __init__(self, voxel_size: float = 0.25, sphere_radius: float = 1.5):
        self.voxel_size = voxel_size
        self.sphere_radius = sphere_radius
        self.car_size = (4.0, 2.0, 6.0)     # 차량 크기
        self.person_size = (1.2, 1.8, 0.8)  # 사람 크기
        
        print(f"🧊 촘촘한 Voxel 시스템 시작")
        print(f"📏 복셀 크기: {voxel_size}m³ (촘촘), 구체 반지름: {sphere_radius}m")
    
    def create_voxels(self, center: Vec3, size: Vec3, is_car: bool = False) -> Set[Tuple[int, int, int]]:
        """중심점 기준 복셀 생성"""
        voxels = set()
        half = (size[0]/2, size[1]/2, size[2]/2)
        
        # 복셀 범위 계산
        min_v = (int((center[0] - half[0]) / self.voxel_size),
                 int((center[1] - half[1]) / self.voxel_size),
                 int((center[2] - half[2]) / self.voxel_size))
        max_v = (int((center[0] + half[0]) / self.voxel_size),
                 int((center[1] + half[1]) / self.voxel_size),
                 int((center[2] + half[2]) / self.voxel_size))
        
        if is_car:
            # 차량: 3단계 현실적 모델 (경량화)
            y_range = max_v[1] - min_v[1] + 1
            middle_y = min_v[1] + y_range // 3
            top_y = min_v[1] + (y_range * 2) // 3
            
            for x in range(min_v[0], max_v[0] + 1):
                for y in range(min_v[1], max_v[1] + 1):
                    for z in range(min_v[2], max_v[2] + 1):
                        # 하단: 바퀴 (앞뒤 + 좌우 끝만)
                        if y < middle_y:
                            wheel_front = abs(z - min_v[2]) <= 1
                            wheel_rear = abs(z - max_v[2]) <= 1
                            wheel_sides = (x == min_v[0] or x == max_v[0])
                            if (wheel_front or wheel_rear) and wheel_sides:
                                voxels.add((x, y, z))
                        
                        # 중단: 차체 본체 (전체)
                        elif y < top_y:
                            voxels.add((x, y, z))
                        
                        # 상단: 루프 (중앙만, 경량화)
                        else:
                            x_margin = max(1, (max_v[0] - min_v[0]) // 4)
                            z_margin = max(1, (max_v[2] - min_v[2]) // 3)
                            if (min_v[0] + x_margin <= x <= max_v[0] - x_margin and
                                min_v[2] + z_margin <= z <= max_v[2] - z_margin):
                                voxels.add((x, y, z))
        else:
            # 사람: 직육면체 (단순)
            for x in range(min_v[0], max_v[0] + 1):
                for y in range(min_v[1], max_v[1] + 1):
                    for z in range(min_v[2], max_v[2] + 1):
                        voxels.add((x, y, z))
        
        return voxels
    
    def voxel_to_world(self, voxel: Tuple[int, int, int]) -> Vec3:
        """복셀을 월드 좌표로 변환"""
        return (voxel[0] * self.voxel_size + self.voxel_size/2,
                voxel[1] * self.voxel_size + self.voxel_size/2,
                voxel[2] * self.voxel_size + self.voxel_size/2)
    
    def calculate_overlap(self, car_voxels: Set, person_voxels: Set) -> Tuple[float, int, int]:
        """복셀 간 겹침 계산 (70% 겹침 시 100% 확률)"""
        if not car_voxels or not person_voxels:
            return 0.0, 0, 0
        
        overlap_pairs = 0
        overlapping_person = set()
        
        # 구체 겹침 검사
        for car_v in car_voxels:
            car_world = self.voxel_to_world(car_v)
            for person_v in person_voxels:
                person_world = self.voxel_to_world(person_v)
                
                distance = math.sqrt(sum((car_world[i] - person_world[i])**2 for i in range(3)))
                
                if distance < (self.sphere_radius * 2):
                    overlap_pairs += 1
                    overlapping_person.add(person_v)
        
        # 확률 계산: 70% 겹침에서 100% 확률
        overlap_count = len(overlapping_person)
        if overlap_count == 0:
            probability = 0.0
        else:
            ratio = overlap_count / len(person_voxels)
            
            # 새로운 공식: 첫 접촉 50%, 70% 겹침에서 100%
            if ratio >= 0.7:  # 70% 이상 겹침
                probability = 100.0
            else:
                # 0% → 50%, 70% → 100% 선형 매핑
                probability = 50.0 + (ratio / 0.7) * 50.0
        
        return min(100.0, probability), overlap_pairs, overlap_count
    
    def create_danger_image(self, car_voxels: Set, person_voxels: Set, 
                           car_center: Vec3, person_center: Vec3,
                           actor_id: str, probability: float) -> Optional[str]:
        """위험 상황 3D 이미지 생성"""
        try:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # 복셀 시각화
            if car_voxels:
                car_coords = np.array([self.voxel_to_world(v) for v in car_voxels])
                ax.scatter(car_coords[:, 0], car_coords[:, 2], car_coords[:, 1],
                          c='blue', s=50, alpha=0.7, marker='s', label=f'Car ({len(car_voxels)})')
            
            if person_voxels:
                person_coords = np.array([self.voxel_to_world(v) for v in person_voxels])
                ax.scatter(person_coords[:, 0], person_coords[:, 2], person_coords[:, 1],
                          c='red', s=50, alpha=0.7, marker='o', label=f'Person ({len(person_voxels)})')
            
            # 중심점 표시
            ax.scatter(car_center[0], car_center[2], car_center[1],
                      c='darkblue', s=100, marker='X', label='Car Center')
            ax.scatter(person_center[0], person_center[2], person_center[1],
                      c='darkred', s=100, marker='X', label='Person Center')
            
            # 거리 계산
            distance = math.sqrt(sum((car_center[i] - person_center[i])**2 for i in range(3)))
            
            ax.set_title(f'DANGER - Actor {actor_id}\nRisk: {probability:.1f}% | Distance: {distance:.2f}m',
                        fontsize=10, color='red', weight='bold')
            ax.set_xlabel('X'); ax.set_ylabel('Z'); ax.set_zlabel('Y')
            ax.legend()
            
            # 파일 저장
            timestamp = datetime.now().strftime("%H%M%S")
            filename = f"danger_{timestamp}_{actor_id}.png"
            plt.savefig(filename, dpi=200, bbox_inches='tight')
            plt.close()
            
            return filename
            
        except Exception as e:
            print(f"⚠️ 이미지 생성 오류: {e}")
            return None
    
    def detect_danger(self, car_pos: Vec3, person_pos: Vec3, actor_id: str) -> Dict:
        """위험 감지 메인 함수"""
        # 복셀 생성
        car_voxels = self.create_voxels(car_pos, self.car_size, is_car=True)
        person_voxels = self.create_voxels(person_pos, self.person_size, is_car=False)
        
        # 겹침 계산
        probability, overlap_pairs, overlap_count = self.calculate_overlap(car_voxels, person_voxels)
        
        # 거리 계산
        distance = math.sqrt(sum((car_pos[i] - person_pos[i])**2 for i in range(3)))
        
        return {
            'actor_id': actor_id,
            'car_voxels': len(car_voxels),
            'person_voxels': len(person_voxels),
            'overlap_pairs': overlap_pairs,
            'overlap_count': overlap_count,
            'probability': probability,
            'distance': distance,
            'is_danger': overlap_pairs > 0,
            'car_voxel_set': car_voxels,
            'person_voxel_set': person_voxels
        }

class VoxelMonitor:
    """경량화된 실시간 모니터"""
    
    def __init__(self, car_id: str = "1003", person_id: str = "1001"):
        self.car_id = car_id
        self.person_id = person_id
        self.detector = VoxelHazardDetector()
        self.last_second = None
        
        print("🎯 경량 Voxel 모니터 시작")
        print(f"📍 차량: {car_id}, 사람: {person_id}")
    
    def predict_position(self, data: pd.DataFrame, horizon: float = 3.0) -> Optional[Vec3]:
        """선형 예측"""
        if len(data) < 2:
            return None
        
        try:
            data = data.sort_values('time')
            dt = float(data['time'].iloc[-1] - data['time'].iloc[0])
            if dt <= 0:
                return None
            
            # 속도 계산
            dx = (float(data['x'].iloc[-1]) - float(data['x'].iloc[0])) / dt
            dy = (float(data['y'].iloc[-1]) - float(data['y'].iloc[0])) / dt
            dz = (float(data['z'].iloc[-1]) - float(data['z'].iloc[0])) / dt
            
            # 미래 위치 예측
            last = data.iloc[-1]
            return (float(last['x']) + dx * horizon,
                   float(last['y']) + dy * horizon,
                   float(last['z']) + dz * horizon)
        except:
            return None
    
    def load_data(self) -> Optional[pd.DataFrame]:
        """데이터 로드"""
        try:
            if not os.path.exists("realtime_data.csv"):
                return None
            df = pd.read_csv("realtime_data.csv")
            return df if len(df) > 0 else None
        except:
            return None
    
    def run(self):
        """메인 루프"""
        print("🔍 모니터링 시작 (Ctrl+C로 종료)")
        
        while True:
            try:
                df = self.load_data()
                if df is None:
                    time.sleep(0.2)
                    continue
                
                # 초 단위 처리
                cur_sec = int(df['time'].max())
                if self.last_second == cur_sec:
                    time.sleep(0.1)
                    continue
                self.last_second = cur_sec
                
                # 데이터 필터링
                df['custom_id'] = df['custom_id'].astype(str)
                car_data = df[df['custom_id'] == self.car_id].tail(20)
                person_data = df[df['custom_id'] == self.person_id].tail(20)
                
                if len(car_data) < 2 or len(person_data) < 2:
                    print(f"[{cur_sec}s] 데이터 부족 - 차:{len(car_data)} 사람:{len(person_data)}")
                    continue
                
                # 3초 후 위치 예측
                car_pred = self.predict_position(car_data)
                person_pred = self.predict_position(person_data)
                
                if car_pred is None or person_pred is None:
                    print(f"[{cur_sec}s] 예측 실패")
                    continue
                
                # 위험 감지
                result = self.detector.detect_danger(car_pred, person_pred, self.person_id)
                
                # 결과 출력
                if result['is_danger']:
                    print(f"🚨[위험] {result['actor_id']} | 확률:{result['probability']:.1f}% | "
                          f"거리:{result['distance']:.1f}m | 접촉:{result['overlap_count']}/{result['person_voxels']}")
                    
                    # 위험 시 이미지 생성
                    filename = self.detector.create_danger_image(
                        result['car_voxel_set'], result['person_voxel_set'],
                        car_pred, person_pred, result['actor_id'], result['probability']
                    )
                    if filename:
                        print(f"   📸 이미지: {filename}")
                
                elif result['distance'] <= 15.0:
                    print(f"👀[근접] {result['actor_id']} | 거리:{result['distance']:.1f}m | "
                          f"복셀:{result['car_voxels']}/{result['person_voxels']}")
                else:
                    print(f"✅[안전] {result['actor_id']} | 거리:{result['distance']:.1f}m | "
                          f"복셀:{result['car_voxels']}/{result['person_voxels']}")
                
                time.sleep(0.1)
                
            except KeyboardInterrupt:
                print("\n🛑 모니터 종료")
                break
            except Exception as e:
                print(f"⚠️ 오류: {e}")
                time.sleep(0.2)

def main():
    monitor = VoxelMonitor()
    monitor.run()

if __name__ == "__main__":
    main()