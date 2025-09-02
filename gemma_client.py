import json, requests
import os
import tkinter as tk
import threading
import time
from datetime import datetime

OLLAMA_HOST = "http://localhost:11434"
MODEL = "gemma:2b"

class GemmaGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Gemma Collision Detection")
        self.root.geometry("500x400")
        self.root.configure(bg='#2b2b2b')
        
        self.last_danger_status = "SAFE"
        self.last_danger_color = "green"
        self.last_probability = 0.0
        
        self.create_widgets()
        self.start_update_thread()
    
    def create_widgets(self):
        title = tk.Label(self.root, text="Gemma Collision Detection", 
                        font=("Arial", 16, "bold"), 
                        bg='#2b2b2b', fg='white')
        title.pack(pady=10)
        
        danger_frame = tk.Frame(self.root, bg='#3d3d3d', relief='ridge', bd=2)
        danger_frame.pack(pady=20, padx=20, fill='x')
        
        tk.Label(danger_frame, text="Collision Risk", font=("Arial", 12, "bold"), bg='#3d3d3d', fg='white').pack(pady=5)
        
        self.danger_status = tk.Label(danger_frame, text="SAFE", font=("Arial", 20, "bold"), bg='#3d3d3d', fg='green')
        self.danger_status.pack(pady=5)
        
        self.probability_label = tk.Label(danger_frame, text="Probability: 0.0%", font=("Arial", 12), bg='#3d3d3d', fg='white')
        self.probability_label.pack()
        
        msg_frame = tk.Frame(self.root, bg='#2b2b2b')
        msg_frame.pack(pady=10, padx=20, fill='both', expand=True)
        
        tk.Label(msg_frame, text="Gemma Messages:", font=("Arial", 10), bg='#2b2b2b', fg='white').pack(anchor='w')
        
        self.ai_message = tk.Text(msg_frame, height=8, width=50, font=("Arial", 9), bg='#1e1e1e', fg='yellow', wrap=tk.WORD, state='disabled')
        self.ai_message.pack(fill='both', expand=True, pady=5)
    
    def update_ai_message(self, message):
        self.ai_message.config(state='normal')
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.ai_message.insert(tk.END, f"[{timestamp}] {message}\n")
        self.ai_message.see(tk.END)
        self.ai_message.config(state='disabled')
    
    def start_update_thread(self):
        def update_loop():
            while True:
                try:
                    self.root.after(0, self.check_danger_alerts)
                except:
                    break
                time.sleep(0.5)
        
        self.update_thread = threading.Thread(target=update_loop, daemon=True)
        self.update_thread.start()
    
    def check_danger_alerts(self):
        try:
            if os.path.exists("danger_alert.json"):
                with open("danger_alert.json", "r", encoding='utf-8') as f:
                    data = json.load(f)
                
                base_prob = data.get('collision_probability', 0)
                yolo_info = read_yolo_detections("yolo_stream.jsonl")
                final_prob = apply_class_weighting(base_prob, yolo_info)
                
                self.last_probability = final_prob
                self.probability_label.config(text=f"Probability: {final_prob:.1f}%")
                
                if final_prob >= 70:
                    self.last_danger_status = "DANGER"
                    self.last_danger_color = "red"
                elif final_prob >= 50:
                    self.last_danger_status = "WARNING"
                    self.last_danger_color = "orange"
                else:
                    self.last_danger_status = "CAUTION"
                    self.last_danger_color = "yellow"
                
                self.danger_status.config(text=self.last_danger_status, fg=self.last_danger_color)
                
                gemma_response = analyze_json_with_weighting(data, yolo_info, final_prob)
                self.update_ai_message(f"Gemma: {gemma_response}")
                
                os.remove("danger_alert.json")
                
                self.root.after(2000, self.reset_to_safe)
                
            else:
                self.danger_status.config(text=self.last_danger_status, fg=self.last_danger_color)
                self.probability_label.config(text=f"Probability: {self.last_probability:.1f}%")
                
        except Exception as e:
            pass
    
    def reset_to_safe(self):
        self.last_danger_status = "SAFE"
        self.last_danger_color = "green"
        self.last_probability = 0.0
        self.danger_status.config(text="SAFE", fg="green")
        self.probability_label.config(text="Probability: 0%")
    
    def run(self):
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            pass

def read_yolo_detections(jsonl_file_path: str) -> dict:
    try:
        if not os.path.exists(jsonl_file_path):
            return {"has_child": False, "child_count": 0}
        
        latest_detection = None
        with open(jsonl_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    detection = json.loads(line)
                    if detection.get('detections'):
                        latest_detection = detection
        
        if not latest_detection:
            return {"has_child": False, "child_count": 0}
        
        child_count = 0
        for detection in latest_detection.get('detections', []):
            if detection.get('class_name') == 'child':
                child_count += 1
        
        return {
            "has_child": child_count > 0,
            "child_count": child_count,
            "time": latest_detection.get('time', 0),
            "frame_idx": latest_detection.get('frame_idx', 0)
        }
        
    except Exception as e:
        return {"has_child": False, "child_count": 0}

def apply_class_weighting(base_probability: float, yolo_info: dict) -> float:
    adjusted_prob = base_probability
    
    if yolo_info.get("has_child", False):
        child_count = yolo_info.get("child_count", 1)
        weight = min(10.0 * child_count, 20.0)
        adjusted_prob += weight
    
    return min(adjusted_prob, 99.9)

def analyze_json_with_weighting(data: dict, yolo_info: dict, final_prob: float) -> str:
    try:
        distance = data.get('distance', 0)
        status = data.get('status', 'UNKNOWN')
        
        if final_prob >= 70:
            system_prompt = (
                "You analyze collision risk data with HIGH DANGER level. "
                "Output format: 'Final collision probability is X.X%. You should to Stop! Critical Danger!' "
                "Use strong warning language for critical situations. "
                "Maximum probability is 99.9%. Never exceed this limit."
            )
        elif final_prob >= 50:
            system_prompt = (
                "You analyze collision risk data with MEDIUM DANGER level. "
                "Output format: 'Final collision probability is X.X%. You need to Stop! Warning.' "
                "Use moderate warning language for dangerous situations. "
                "Maximum probability is 99.9%. Never exceed this limit."
            )
        else:
            system_prompt = (
                "You analyze collision risk data with LOW DANGER level. "
                "Output format: 'Final collision probability is X.X%. Caution advised.' "
                "Use cautious language for low risk situations. "
                "Maximum probability is 99.9%. Never exceed this limit."
            )
        
        weight_info = ""
        if yolo_info.get("has_child"):
            weight_info = f" (Child detected: +{final_prob - data.get('collision_probability', 0):.1f}% weight)"
        
        user_input = f"Final probability: {final_prob}%, Distance: {distance:.1f}m, Status: {status}{weight_info}"

        payload = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            "stream": False,
            "options": {"temperature": 0.1}
        }

        res = requests.post(f"{OLLAMA_HOST}/api/chat",
                           headers={"Content-Type": "application/json"},
                           data=json.dumps(payload), timeout=10)
        res.raise_for_status()
        
        response = res.json()
        gemma_response = response.get("message", {}).get("content", "").strip()
        
        if yolo_info.get("has_child"):
            gemma_response += f"\nChild Protection: +{final_prob - data.get('collision_probability', 0):.1f}% risk weight applied"
        
        return gemma_response
        
    except Exception as e:
        if final_prob >= 70:
            warning = f"Brake! Brake! Brake!\n🚨 You have to Stop! High Danger!\nCollision probability is {final_prob:.1f}%."
        else:
            warning = f"Brake! Brake! Brake!\n⚠️ You need to Stop! Warning.\nCollision probability is {final_prob:.1f}%."
        
        if yolo_info.get("has_child"):
            warning += f"\nChild detected! Extra caution required."
            
        return warning

def main():
    gui = GemmaGUI()
    gui.run()

if __name__ == "__main__":
    main()