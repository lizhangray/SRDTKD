# test demo

# infer for DTKD-RFDN in Set5
python main.py --Train False --model_name RFDN --checkpoint DTKD-RFDN.pth --test_folder Datasets2023/GTmod12_LRx4/Set5_LRbicx4

# infer for DTKD-LBNet in Set5
python main.py --Train False --model_name LBNet --checkpoint DTKD-LBNet.pth --test_folder Datasets2023/GTmod12_LRx4/Set5_LRbicx4


