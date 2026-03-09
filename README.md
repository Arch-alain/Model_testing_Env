# 🔬 Malaria Detection: Edge Benchmarking Suite

This repository contains a containerized environment designed to evaluate different models including **YOLO** variants model, specifically optimized for **Raspberry Pi 4 and 5**.

The project is Dockerized to bypass the common package incompatibility issues often encountered when installing `ultralytics`, `torch`, and `OpenCV` directly on Raspberry Pi OS.

---

## ⚙️ Prerequisites

* **Hardware:** Raspberry Pi 4 (4GB+) or Raspberry Pi 5.
* **OS:** Raspberry Pi OS **64-bit**.
* **Storage:** At least 32GB of free space for the Docker image and layers.

---

## 🚀 Getting Started

### 1. Install Docker on the Raspberry Pi
If Docker is not yet installed on your Pi, run the following commands:

```bash
curl -fsSL [https://get.docker.com](https://get.docker.com) -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ${USER}
Note: Log out and log back in (or reboot) for the permissions to take effect.
```
### 2. Securely Transfer Test images 
Due to sensitivity issues of our dataset. Do not upload images to GitHub. Use scp (Secure Copy) to move your private test images and trained models from your local machine directly to the Pi.

Run these on your LOCAL machine:

``` bash
# Create the directory structure on the Pi
ssh username@<your_pi_ip> "mkdir -p ~/malaria_bench/test_data ~/malaria_bench/models ~/malaria_bench/results"
# Transfer your private images
scp -r ./local_test_images/* username@<your_pi_ip>:~/malaria_bench/test_data/

# Transfer your trained YOLO model (.pt file)
scp ./yolov10_nano.pt username@<your_pi_ip>:~/malaria_bench/models/
```
### 3. Build & Run the Benchmark
Navigate to the project folder on your Raspberry Pi and execute the build:

Build the Image:

``` bash
docker build -t malaria-model-tester .
```
Run the Inference Session:
We use Docker Volumes to map the Pi's local folders to the container. This keeps your private data on the physical disk and ensures results are saved permanently.

``` bash
docker run --rm \
  -v $(pwd)/models:/usr/src/app/models \
  -v $(pwd)/test_data:/usr/src/app/test_data \
  -v $(pwd)/results:/usr/src/app/results \
  malaria-model-tester