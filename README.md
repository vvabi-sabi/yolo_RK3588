# yolov8_RK3588

## Install
Install miniconda
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh
bash Miniconda3-latest-Linux-aarch64.sh
```
Create conda python3.9 - env
```
conda create -n <env-name> python=3.9
```
Activate virtualenv
```
conda activate <env-name>
```
Install RKNN-Toolkit2-Lite
```
cd yolact/install/
pip install -r requirements.txt
pip install rknn_toolkit_lite2-1.4.0-cp39-cp39-linux_aarch64.whl
```

## Run
```
git clone https://github.com/vvabi-sabi/yolact_RK3588.git
cd yolo_RK3588
python main.py
```