python main.py -d faceforensics -m train -mn xception
Start-Sleep -s 30
python main.py -d faceforensics -m train -mn inception_v3
Start-Sleep -s 30
python main.py -d faceforensics -m train -mn efficientnet-b5
Start-Sleep -s 30
python main.py -d faceforensics -m train -mn resnet152
Start-Sleep -s 30
python main.py -d faceforensics -m train -mn resnext101