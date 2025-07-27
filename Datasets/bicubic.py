import torch
import glob
import cv2
import PIL.Image as Image
from pathlib import Path
import torchvision

# images = glob.glob("manga109/*")
# 将urban100文件夹下的所有图片进行bicubic操作
# images = glob.glob("urban100/*")
# images = glob.glob("D:\Desktop\manga109\*")

Totensor = torchvision.transforms.ToTensor()
ToImage  = torchvision.transforms.ToPILImage()
'''
for img in images:
    name = img.replace('BSD100/','BSD100_LR/x4/').replace('.png','x4.png')
    print(img)
    img = Image.open(img)
    img = Totensor(img).unsqueeze(0)
    img = torch.nn.functional.interpolate(img, scale_factor=0.25, mode='bicubic',align_corners=True, recompute_scale_factor=True).clamp(min=0, max=255)
    img = ToImage(img.squeeze())
    img.save(name)

# LR = torch.nn.functional.interpolate(HR, scale_factor=0.5) # , mode='bicubic')
'''
def bicubic_scaling(input_dir, scales):
    input_path = Path(input_dir)
    scale_paths = {}
    for scale in scales:
        parent_path = (input_path.parent / f'{input_path.name}_LR/x{scale}')
        scale_paths[scale] = parent_path
        parent_path.mkdir(parents=True, exist_ok=True)
    for file_name in input_path.glob('*.png'):
        hr = Image.open(file_name).convert('RGB')
        for scale, scale_path in scale_paths.items():
            lr = hr.resize((int(hr.width / scale), int(hr.height/ scale)), resample=Image.BICUBIC)
            lr.save(scale_path / f'{file_name.name}'.replace('.png','x4.png'), "PNG")

# bicubic_scaling('./manga109', [4])
# bicubic_scaling('./urban100', [4])
if __name__ == '__main__':
    # 将D:\Desktop\manga109文件夹下的所有图片进行4倍bicubic操作
    # 会生成manga109_LR文件夹
    bicubic_scaling('D:\Desktop\manga109', [4])
    print("Successfully run!")