import os
import time
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

try:
    import tensorflow as tf
    import joblib
    from sklearn.preprocessing import MinMaxScaler
    tf.get_logger().setLevel('ERROR')
    TENSORFLOW_AVAILABLE = True
    print("TensorFlow available - Pre-trained LSTM mode")
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not found - using linear prediction only")

class PretrainedLSTMPredictor:
    def __init__(self, sequence_length=5, prediction_seconds=3):
        self.data_file = "realtime_data.csv"
        self.processed_seconds = set()
        self.sequence_length = sequence_length
        self.prediction_seconds = prediction_seconds
        
        self.vehicle_data = []
        self.person_data = []
        self.max_data_size = 150
        
        self.vehicle_model = None
        self.person_model = None
        self.vehicle_scaler = None
        self.person_scaler = None
        
        self.load_pretrained_models()
        
        print("Pre-trained LSTM predictor started")
        print(f"Sequence length: {sequence_length} seconds")
        print(f"Prediction time: {prediction_seconds} seconds ahead")
        print("=" * 50)
    
    def load_pretrained_models(self):
        if not TENSORFLOW_AVAILABLE:
            print("TensorFlow not available - models not loaded")
            return
        
        try:
            if os.path.exists("models/vehicle_lstm_model.h5") and os.path.exists("models/vehicle_scaler.pkl"):
                self.vehicle_model = tf.keras.models.load_model("models/vehicle_lstm_model.h5")
                self.vehicle_scaler = joblib.load("models/vehicle_scaler.pkl")
                print("✓ Vehicle LSTM model loaded")
            else:
                print("✗ Vehicle model not found")
                
            if os.path.exists("models/person_lstm_model.h5") and os.path.exists("models/person_scaler.pkl"):
                self.person_model = tf.keras.models.load_model("models/person_lstm_model.h5")
                self.person_scaler = joblib.load("models/person_scaler.pkl")
                print("✓ Person LSTM model loaded")
            else:
                print("✗ Person model not found")
                
        except Exception as e:
            print(f"Model loading error: {e}")
            self.vehicle_model = None
            self.person_model = None
    
    def lstm_predict(self, data, model, scaler, name):
        if model is None or scaler is None:
            return self.linear_predict(data, name)
        
        if len(data) < self.sequence_length:
            return self.linear_predict(data, name)
        
        try:
            recent_data = data[-self.sequence_length:]
            df = pd.DataFrame(recent_data).sort_values('time')
            coords = df[['x', 'y', 'z']].values.astype(np.float32)
            
            coords_scaled = scaler.transform(coords)
            X = coords_scaled.reshape(1, self.sequence_length, 3)
            
            current_sequence = X[0]
            
            for step in range(self.prediction_seconds):
                next_pred = model.predict(current_sequence.reshape(1, self.sequence_length, 3), verbose=0)
                current_sequence = np.vstack([current_sequence[1:], next_pred[0]])
            
            final_pred = scaler.inverse_transform(next_pred)[0]
            
            return {
                'pred_x': float(final_pred[0]),
                'pred_y': float(final_pred[1]),
                'pred_z': float(final_pred[2]),
                'method': 'LSTM'
            }
            
        except Exception as e:
            return self.linear_predict(data, name)
    
    def linear_predict(self, data, name):
        if len(data) < 2:
            return None
        
        try:
            recent = data[-min(5, len(data)):]
            df = pd.DataFrame(recent).sort_values('time')
            
            if len(df) < 2:
                return None
            
            time_diff = df['time'].iloc[-1] - df['time'].iloc[0]
            if time_diff <= 0:
                time_diff = 1.0
            
            x_speed = (df['x'].iloc[-1] - df['x'].iloc[0]) / time_diff
            y_speed = (df['y'].iloc[-1] - df['y'].iloc[0]) / time_diff
            z_speed = (df['z'].iloc[-1] - df['z'].iloc[0]) / time_diff
            
            last_pos = df.iloc[-1]
            pred_x = last_pos['x'] + x_speed * self.prediction_seconds
            pred_y = last_pos['y'] + y_speed * self.prediction_seconds
            pred_z = last_pos['z'] + z_speed * self.prediction_seconds
            
            return {
                'pred_x': float(pred_x),
                'pred_y': float(pred_y),
                'pred_z': float(pred_z),
                'method': 'linear_prediction'
            }
            
        except Exception as e:
            return None
    
    def get_data_by_second(self):
        try:
            if not os.path.exists(self.data_file):
                return None
            
            df = pd.read_csv(self.data_file)
            if len(df) == 0:
                return None
            
            max_time = df['time'].max()
            current_second = int(max_time)
            
            if current_second in self.processed_seconds:
                return None
            
            self.processed_seconds.add(current_second)
            
            second_data = df[(df['time'] >= current_second) & (df['time'] < current_second + 1)]
            
            if len(second_data) == 0:
                return None
            
            second_data['custom_id'] = second_data['custom_id'].astype(str)
            
            vehicle_data = second_data[second_data['custom_id'] == '1003']
            person_data = second_data[second_data['custom_id'] == '1001']
            
            for _, row in vehicle_data.iterrows():
                self.vehicle_data.append({
                    'time': float(row['time']),
                    'x': float(row['x']),
                    'y': float(row['y']),
                    'z': float(row['z'])
                })
            
            for _, row in person_data.iterrows():
                self.person_data.append({
                    'time': float(row['time']),
                    'x': float(row['x']),
                    'y': float(row['y']),
                    'z': float(row['z'])
                })
            
            if len(self.vehicle_data) > self.max_data_size:
                self.vehicle_data = self.vehicle_data[-self.max_data_size:]
            if len(self.person_data) > self.max_data_size:
                self.person_data = self.person_data[-self.max_data_size:]
            
            return {
                'second': current_second,
                'vehicle_count': len(vehicle_data),
                'person_count': len(person_data)
            }
            
        except Exception as e:
            return None
    
    def run(self):
        print("Pre-trained LSTM predictor running... (Ctrl+C to exit)")
        
        while True:
            try:
                data = self.get_data_by_second()
                if not data:
                    time.sleep(0.1)
                    continue
                
                current_second = data['second']
                target_time = current_second + self.prediction_seconds
                
                print(f"[{current_second}s] Vehicle:{len(self.vehicle_data)}, Person:{len(self.person_data)}")
                
                vehicle_result = None
                person_result = None
                
                if len(self.vehicle_data) >= 2:
                    vehicle_result = self.lstm_predict(
                        self.vehicle_data, self.vehicle_model, self.vehicle_scaler, "Vehicle"
                    )
                
                if len(self.person_data) >= 2:
                    person_result = self.lstm_predict(
                        self.person_data, self.person_model, self.person_scaler, "Person"
                    )
                
                print(f"[{target_time}s prediction]")
                
                if person_result:
                    print(f"Person: ({person_result['pred_x']:.1f}, {person_result['pred_y']:.1f}, {person_result['pred_z']:.1f}) [{person_result['method']}]")
                else:
                    print("Person: prediction unavailable")
                
                if vehicle_result:
                    print(f"Vehicle: ({vehicle_result['pred_x']:.1f}, {vehicle_result['pred_y']:.1f}, {vehicle_result['pred_z']:.1f}) [{vehicle_result['method']}]")
                else:
                    print("Vehicle: prediction unavailable")
                
                print("-" * 60)
                
            except KeyboardInterrupt:
                print("\nTerminated")
                break
            except Exception as e:
                time.sleep(0.1)

def main():
    predictor = PretrainedLSTMPredictor()
    predictor.run()

if __name__ == "__main__":
    main()