# CMUnet
###introduction
CMUnet is a new Unet model based on depth separable convolution. CMUnet has achieved good segmentation results on BUSI dataset and thyroid dataset we collected. This project is its pytorch implementation.
###datasets
Please store the BUSI dataset or your own dataset in the following directory architecture. This implementation only supports single class segmentation.
'''
├── CMUnet
    ├── inputs
        ├── BUSI
            ├── images
            |   ├── 0a7e06.jpg
            │   ├── 0aab0a.jpg
            │   ├── 0b1761.jpg
            │   ├── ...
            |
            └── masks
                ├── 0
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
        ├── your dataset name
            ├── images
            |   ├── 0a7e06.jpg
            │   ├── 0aab0a.jpg
            │   ├── 0b1761.jpg
            │   ├── ...
            |
            └── masks
                ├── 0
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
'''
###training
Create "checkpoint" directory under the CMUnet directory to save the pth file.

'''
nohup python train.py --dataset BUSI --name CMUnet  --img_ext .png --mask_ext .png --lr 0.0001 --epochs 400 --input_w 256 --input_h 256 --b 8 > log/CMUnet.log 2>&1 &
'''








