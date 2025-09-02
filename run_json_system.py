import subprocess
import time
import sys
import signal
import os

class JsonSystemRunner:
    def __init__(self):
        self.processes = []
        self.running = True

    def start_process(self, script: str, name: str) -> bool:
        if not os.path.exists(script):
            return False

        try:
            process = subprocess.Popen([sys.executable, script])
            self.processes.append((name, process))
            return True
        except Exception as e:
            return False

    def cleanup(self):
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

    def signal_handler(self, signum, frame):
        self.running = False

    def run(self):
        signal.signal(signal.SIGINT, self.signal_handler)
        
        if not self.start_process('udp_receiver.py', 'UDP receiver'):
            return
        time.sleep(2)
        
        self.start_process('lstm_predictor.py', 'LSTM predictor')
        time.sleep(2)
        
        if not self.start_process('voxel_hazard_clean.py', 'Voxel + JSON'):
            self.cleanup()
            return
        
        if not self.start_process('gemma_client.py', 'Gemma GUI'):
            self.cleanup()
            return

        try:
            while self.running:
                time.sleep(2)
                
                dead = [(n, p.poll()) for n, p in self.processes if p.poll() is not None]
                
        except KeyboardInterrupt:
            pass
        finally:
            self.cleanup()

def main():
    runner = JsonSystemRunner()
    runner.run()

if __name__ == "__main__":
    main()