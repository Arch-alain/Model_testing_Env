import os
import time
import json
import platform
import psutil
from datetime import datetime
from dataclasses import dataclass, asdict
from ultralytics import YOLO

# 1. The "Research Object" for Reproducibility
@dataclass
class InferenceResult:
    filename: str
    inference_time_ms: float
    preprocess_time_ms: float
    postprocess_time_ms: float
    detections_count: int

@dataclass
class BenchmarkSession:
    session_id: str
    timestamp: str
    model_name: str
    hardware: dict
    test_set_version: str
    results: list[InferenceResult]
    avg_latency_ms: float = 0.0

    def save(self, folder="results"):
        os.makedirs(folder, exist_ok=True)
        filepath = os.path.join(folder, f"bench_{self.session_id}.json")
        with open(filepath, 'w') as f:
            json.dump(asdict(self), f, indent=4)
        print(f"✅ Session saved to {filepath}")

# 2. The Benchmarking Engine
class MalariaBenchmarker:
    def __init__(self, model_path, test_data_path):
        self.model = YOLO(model_path)
        self.data_path = test_data_path
        self.model_name = os.path.basename(model_path)

    def get_sys_info(self):
        return {
            "os": platform.system(),
            "processor": platform.processor(),
            "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "python_v": platform.python_version()
        }

    def run(self, version_tag="v1.0-alpha"):
        print(f"🚀 Starting Benchmarking for {self.model_name}...")
        
        images = [f for f in os.listdir(self.data_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
        if not images:
            print("❌ No images found in test_data folder!")
            return

        session = BenchmarkSession(
            session_id=datetime.now().strftime("%Y%m%d_%H%M%S"),
            timestamp=datetime.now().isoformat(),
            model_name=self.model_name,
            hardware=self.get_sys_info(),
            test_set_version=version_tag,
            results=[]
        )

        # WARM-UP (Crucial for accurate benchmarking)
        print("🔥 Warming up model...")
        self.model.predict(os.path.join(self.data_path, images[0]), verbose=False)

        # INFERENCE LOOP
        total_inf_time = 0
        for img_name in images:
            img_path = os.path.join(self.data_path, img_name)
            
            # Run Inference
            results = self.model.predict(img_path, verbose=False)[0]
            
            # Extract Timings (in ms)
            metrics = InferenceResult(
                filename=img_name,

                detections_count=len(results.boxes)
            )
            
            session.results.append(metrics)
            total_inf_time += metrics.inference_time_ms
            print(f"Processed {img_name}: {metrics.inference_time_ms:.2f}ms")

        session.avg_latency_ms = total_inf_time / len(images)
        session.save()
        print(f"📊 Benchmark Complete. Avg Latency: {session.avg_latency_ms:.2f}ms")

if __name__ == "__main__":
    # Update these paths to your actual files
    MODEL_FILE = "/usr/src/app/models/yolov10_nano.pt" # Or your custom malaria model
    DATA_DIR = "test_data/"
    
    tester = MalariaBenchmarker(MODEL_FILE, DATA_DIR)
    tester.run(version_tag="lab_test_mac_01")