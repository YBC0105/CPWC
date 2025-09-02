import subprocess
import time
import sys
import signal
import os

class VoxelRunner:
    def __init__(self):
        self.processes = []
        self.running = True

    def start_process(self, script: str, name: str) -> bool:
        """프로세스 시작"""
        if not os.path.exists(script):
            print(f"❌ {name}: '{script}' 파일 없음")
            return False

        try:
            print(f"🚀 {name} 시작...")
            env = os.environ.copy()
            env['PYTHONDONTWRITEBYTECODE'] = '1'
            
            process = subprocess.Popen([sys.executable, '-B', script], env=env)
            self.processes.append((name, process))
            print(f"✅ {name} 시작 (PID: {process.pid})")
            return True
        except Exception as e:
            print(f"❌ {name} 실패: {e}")
            return False

    def cleanup(self):
        """모든 프로세스 종료"""
        print("\n🛑 시스템 종료...")
        for name, process in self.processes:
            if process and process.poll() is None:
                try:
                    process.terminate()
                    process.wait(timeout=3)
                except:
                    try:
                        process.kill()
                    except:
                        pass
                print(f"   {name} 종료")
        print("✅ 종료 완료")

    def signal_handler(self, signum, frame):
        self.running = False

    def check_files(self) -> bool:
        """필수 파일 확인"""
        files = ['udp_receiver.py', 'lstm_predictor.py', 'voxel_hazard_view.py']
        missing = [f for f in files if not os.path.exists(f)]
        
        if missing:
            print("❌ 필수 파일 없음:", missing)
            return False
        
        # matplotlib 확인
        try:
            import matplotlib
            print("✅ 의존성 확인 완료")
            return True
        except ImportError:
            print("❌ matplotlib 필요: pip install matplotlib")
            return False

    def run(self):
        """메인 실행"""
        signal.signal(signal.SIGINT, self.signal_handler)
        
        print("경량 Voxel 위험 감지 시스템")
        print("=" * 40)
        
        if not self.check_files():
            return
        
        # 순차 시작
        if not self.start_process('udp_receiver.py', 'UDP 수신기'):
            return
        time.sleep(2)
        
        if not self.start_process('lstm_predictor.py', 'LSTM 예측기'):
            self.cleanup()
            return
        time.sleep(2)
        
        if not self.start_process('voxel_hazard_view.py', 'Voxel 위험감지'):
            print("⚠️ 위험감지 실패")
        
        print("\n🎯 시스템 실행 중!")
        print("📋 기능:")
        print("   • 실시간 데이터 수집")
        print("   • 3초 후 위치 예측") 
        print("   • Voxel 위험 감지")
        print("   • 위험 시 PNG 생성")
        print("\n⚡ Ctrl+C로 종료")
        print("=" * 40)
        
        # 모니터링
        try:
            while self.running:
                time.sleep(2)
                
                # 프로세스 상태 확인
                dead = [(n, p.poll()) for n, p in self.processes if p.poll() is not None]
                
                if dead:
                    for name, code in dead:
                        print(f"⚠️ {name} 종료 (코드: {code})")
                        if name in ('UDP 수신기', 'LSTM 예측기'):
                            self.running = False
                            break
                
                if all(p.poll() is not None for _, p in self.processes):
                    break
                    
        except KeyboardInterrupt:
            pass
        finally:
            self.cleanup()

def main():
    runner = VoxelRunner()
    runner.run()

if __name__ == "__main__":
    main()