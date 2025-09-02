import socket
import time
import os
from datetime import datetime

class UDPReceiver:
    def __init__(self, ip='10.3.120.104', port=4000):
        self.ip = ip
        self.port = port
        self.sock = None
        self.running = False
        
        self.data_file = "realtime_data.csv"
        self.log_file = "receiver_log.txt"
        
        self.init_files()
    
    def init_files(self):
        if os.path.exists(self.data_file):
            os.remove(self.data_file)
        
        with open(self.data_file, "w", encoding='utf-8') as f:
            f.write("time,custom_id,x,y,z\n")
        
        with open(self.log_file, "w", encoding='utf-8') as f:
            f.write(f"UDP receiver started: {datetime.now()}\n")
    
    def start_udp_listener(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2097152)
            self.sock.bind((self.ip, self.port))
            self.running = True
            
            packet_count = 0
            
            while self.running:
                try:
                    self.sock.settimeout(1.0)
                    data, addr = self.sock.recvfrom(65536)
                    
                    packet_count += 1
                    self.process_data(data.decode('utf-8'), packet_count)
                    
                except socket.timeout:
                    continue
                except Exception as e:
                    self.log_error(f"UDP receive error: {e}")
                    
        except Exception as e:
            self.log_error(f"UDP socket creation error: {e}")
    
    def process_data(self, data_string, packet_count):
        lines = data_string.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or 'time,position' in line:
                continue
            
            parsed_data = self.parse_csv_line(line)
            if parsed_data:
                self.save_data(parsed_data)
    
    def parse_csv_line(self, csv_string):
        try:
            parts = csv_string.split(',')
            
            if len(parts) >= 12:
                time_val = self.safe_float(parts[0])
                custom_id = parts[7].strip()
                x_val = self.safe_float(parts[9])
                y_val = self.safe_float(parts[10])
                z_val = self.safe_float(parts[11])
                
                if abs(x_val) > 0.1 or abs(y_val) > 0.1 or abs(z_val) > 0.1:
                    if custom_id in ['1001', '1003']:
                        return {
                            'time': time_val,
                            'custom_id': custom_id,
                            'x': x_val,
                            'y': y_val,
                            'z': z_val
                        }
        except Exception as e:
            self.log_error(f"Parse error: {e}")
        
        return None
    
    def save_data(self, data):
        try:
            with open(self.data_file, "a", encoding='utf-8') as f:
                line = f"{data['time']:.2f},{data['custom_id']},{data['x']:.2f},{data['y']:.2f},{data['z']:.2f}\n"
                f.write(line)
                f.flush()
        except Exception as e:
            self.log_error(f"File save error: {e}")
    
    def safe_float(self, value, default=0.0):
        try:
            if value is None or str(value).strip() == '':
                return default
            return float(str(value).strip())
        except (ValueError, TypeError):
            return default
    
    def log_error(self, message):
        try:
            with open(self.log_file, "a", encoding='utf-8') as f:
                f.write(f"{datetime.now()}: {message}\n")
        except:
            pass
    
    def stop(self):
        self.running = False
        if self.sock:
            self.sock.close()
        
        with open(self.log_file, "a", encoding='utf-8') as f:
            f.write(f"UDP receiver stopped: {datetime.now()}\n")

def main():
    receiver = UDPReceiver()
    
    try:
        receiver.start_udp_listener()
    except KeyboardInterrupt:
        receiver.stop()

if __name__ == "__main__":
    main()