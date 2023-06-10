from mmpretrain import ImageClassificationInferencer
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

inf=ImageClassificationInferencer(r'F:\openmmlab\mmpretrain-main\fruits\fruits_config.py',pretrained=r'F:\openmmlab\mmpretrain-main\fruits\epoch_50.pth')
img_path=r'C:\Users\17219\Desktop\xiangjiao.jpg'
res=inf(img_path,show=True)
print(res)
pred_label=res[0]['pred_label']
pred_score=res[0]['pred_score']
pred_class=res[0]['pred_class']
print(pred_label,pred_score,pred_class)
img2=cv2.imread(img_path)
# cv2.putText(img2, u'label:{} socre:{:.2f} class:{}'.format(pred_label,pred_score,pred_class), (100, 250), font, 1, (255, 255, 255), 2, cv2.LINE_AA,fontFile=font_file)
img = Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
draw = ImageDraw.Draw(img)
fontText = ImageFont.truetype("E:\anaconda\envs\OPENMMLAB\Lib\site-packages\matplotlib\mpl-data\fonts\ttf\msyh.ttc", 30, encoding="utf-8")
draw.text((50,50 ), 'label:{} socre:{:.2f} class:{}'.format(pred_label,pred_score,pred_class), (255,255,255), font=fontText)

img = np.asarray(img)
img =cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

new_path=img_path.split('/')[-1]
cv2.imwrite(f'outputs/{new_path}',img)

plt.imshow(img)