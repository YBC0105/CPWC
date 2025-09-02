import subprocess
import time
import sys
import signal
import os
from typing import List, Tuple, Optional

class VoxelSystemRunner:
    def __init__(self):
        self.processes: List[Tuple[str, subprocess.Popen]] = []
        self.running = True

    def start_process(self, script_name: str, description: str, args: Optional[List[str]] = None) -> bool:
        """서브프로세스 시작 (스크립트 존재 확인 후 실행)"""
        if not os.path.exists(script_name):
            print(f"[경고] {description}: 파일 '{script_name}' 이(가) 없어서 건너뜀")
            return False

        try:
            print(f"{description} 시작 중... ({script_name})")
            # Python 캐시 생성 방지를 위한 환경변수 및 -B 옵션 추가
            env = os.environ.copy()
            env['PYTHONDONTWRITEBYTECODE'] = '1'
            
            cmd = [sys.executable, '-B', script_name] + (args or [])
            process = subprocess.Popen(cmd, env=env)
            self.processes.append((description, process))
            print(f"{description} 시작됨 (PID: {process.pid})")
            return True
        except Exception as e:
            print(f"{description} 시작 실패: {e}")
            return False

    def cleanup(self):
        """모든 하위 프로세스 종료"""
        print("\n🛑 Voxel 시스템 종료 중...")
        for name, process in self.processes:
            if process and process.poll() is None:
                print(f"{name} 종료 중...(PID: {process.pid})")
                try:
                    process.terminate()
                except Exception:
                    pass
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    print(f"{name} 강제 종료...(PID: {process.pid})")
                    try:
                        process.kill()
                    except Exception:
                        pass
                print(f"{name} 종료됨")
        self.processes.clear()

    def _handle_signal(self, signum, frame):
        """시그널 핸들러"""
        self.running = False

    def _install_signal_handlers(self):
        """SIGINT/가능하면 SIGTERM 핸들링"""
        signal.signal(signal.SIGINT, self._handle_signal)
        try:
            signal.signal(signal.SIGTERM, self._handle_signal)
        except Exception:
            pass

    def check_required_files(self) -> bool:
        """필수 파일들이 존재하는지 확인"""
        required_files = [
            'udp_receiver.py',
            'lstm_predictor.py', 
            'voxel_hazard.py'
        ]
        
        missing_files = []
        for file in required_files:
            if not os.path.exists(file):
                missing_files.append(file)
        
        if missing_files:
            print("❌ 다음 필수 파일들이 없습니다:")
            for file in missing_files:
                print(f"   - {file}")
            print("\n모든 파일이 같은 폴더에 있는지 확인하세요.")
            return False
        
        print("✅ 모든 필수 파일 확인 완료")
        return True

    def print_system_info(self):
        """시스템 정보 출력"""
        print("Voxel 기반 실시간 위치 예측 & 위험 감지 시스템")
        print("=" * 60)
        print("실행할 구성 요소:")
        print("   1. UDP 수신기     → 시뮬레이터 데이터 실시간 수집")
        print("   2. LSTM 예측기    → 딥러닝 기반 3초 후 위치 예측")
        print("   3. Voxel 위험감지 → 현실적 형태 기반 충돌 위험 판단")
        print("혁신 기술: 해부학적 Voxel 모델링 vs 기존 바운딩 박스")
        print("=" * 60)

    def run(self):
        """메인 실행 루프"""
        self._install_signal_handlers()
        self.print_system_info()

        # 필수 파일 확인
        if not self.check_required_files():
            return

        print("\n시스템 시작 순서:")
        
        # 1) UDP 수신기 (데이터 수집 최우선)
        print("\n1. UDP 수신기 시작...")
        if not self.start_process('udp_receiver.py', 'UDP 수신기'):
            print("UDP 수신기를 시작할 수 없습니다. 시스템을 종료합니다.")
            return

        print("UDP 수신기 안정화 대기 (3초)...")
        time.sleep(3)

        # 2) LSTM 예측기 (예측 시스템)
        print("\n2. LSTM 예측기 시작...")
        if not self.start_process('lstm_predictor.py', 'LSTM 예측기'):
            print("LSTM 예측기를 시작할 수 없습니다. 시스템을 종료합니다.")
            self.cleanup()
            return

        print("LSTM 예측기 안정화 대기 (2초)...")
        time.sleep(2)

        # 3) Voxel 위험 감지 (모니터링 시스템)
        print("\n3. 현실적 Voxel 위험 감지 시작...")
        if not self.start_process('voxel_hazard.py', '현실적 Voxel 위험감지'):
            print("Voxel 위험감지를 시작하지 못했습니다. (UDP+LSTM만 실행)")

        print("\n" + "="*60)
        print("현실적 Voxel 시스템 실행 중!")
        
        # 실행 중인 구성요소 확인
        running_components = []
        for name, process in self.processes:
            if process.poll() is None:
                running_components.append(name)
        
        if running_components:
            print("실행 중인 구성요소:")
            for i, component in enumerate(running_components, 1):
                print(f"   {i}. {component}")
        
        print("\n이제 시뮬레이터를 실행하여 데이터를 전송하세요!")
        print("Unity 시뮬레이터 → UDP 전송 시작")
        print("터미널에서 실시간 현실적 Voxel 위험 감지 결과를 확인하세요")
        print("차량: 3단계 모델 (바퀴/차체/루프)")  
        print("사람: 4단계 모델 (다리/몸통/머리)")
        print("vs 바운딩 박스: 해부학적 정밀도로 혁신적 충돌 감지")
        print("\nCtrl+C 를 눌러 전체 시스템을 종료할 수 있습니다")
        print("👀 터미널에서 실시간 Voxel 위험 감지 결과를 확인하세요")
        print("\n⚡ Ctrl+C 를 눌러 전체 시스템을 종료할 수 있습니다")
        print("="*60)

        # 상태 모니터링 루프
        try:
            check_count = 0
            while self.running:
                time.sleep(1)
                check_count += 1
                
                # 10초마다 프로세스 상태 확인
                if check_count % 10 == 0:
                    dead_processes = []
                    for name, process in list(self.processes):
                        rc = process.poll()
                        if rc is not None:
                            dead_processes.append((name, rc))
                    
                    if dead_processes:
                        print(f"\n프로세스 상태 변경 감지:")
                        for name, rc in dead_processes:
                            print(f"   {name} 종료됨 (코드: {rc})")
                            
                            # 핵심 프로세스(UDP/LSTM)가 종료되면 시스템 중단
                            if name in ('UDP 수신기', 'LSTM 예측기'):
                                print(f"핵심 구성요소 '{name}' 종료로 인해 시스템을 중단합니다.")
                                self.running = False
                                break
                
                # 모든 프로세스가 종료되었는지 확인
                if all(process.poll() is not None for _, process in self.processes):
                    print("모든 구성요소가 종료되었습니다.")
                    self.running = False
                    break

        except KeyboardInterrupt:
            print("\n사용자가 종료를 요청했습니다.")
        finally:
            self.cleanup()

        print("Voxel 시스템 종료 완료")
        print("시스템을 다시 시작하려면 'python voxel_run.py' 를 실행하세요")

def main():
    """메인 함수"""
    print("환경 설정 중...")
    
    # Python 캐시 문제 방지
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    
    runner = VoxelSystemRunner()
    runner.run()

if __name__ == "__main__":
    main()