import numpy as np
import cv2
#import matplotlib.pyplot as plt


def main():

    x1 = 916
    y1 = 258
    x2 = 916+165
    y2 = 258+473
    # 1.导入图片
    img_src = cv2.imread('/home/ww/catkin_ws/src/fairmot_ros_wzq/scripts/00110.jpg')
    
    # 2.创建掩模图片
    mask = np.zeros(img_src.shape[:2], np.uint8)
    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)

    # 3.执行区域 前景背景分割
    rect = (x1, y1, x2, y2)
    #cv2.grabCut(img_src, mask, rect, bg_model, fg_model, 5, cv2.GC_INIT_WITH_RECT)

    # 5.执行mask 前景背景分割
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if ((i>y1) & (i<y2) & (j>x1) & (j<x2)):
                mask[i,j] = 3
            else:
                mask[i,j] = 0
    #cv2.imshow("img",mask)
    #cv2.waitKey()
    mask, bg_model, fg_model = cv2.grabCut(img_src, mask, rect, bg_model, fg_model, 5, cv2.GC_INIT_WITH_MASK)

    mask = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")

    # 6.分离前景背景图片
    img_gc = img_src * mask[:, :, np.newaxis]
    img_gc = cv2.cvtColor(img_gc, cv2.COLOR_BGR2RGB)

    # 7.显示结果
  #  plt.figure("显示结果", figsize=(12, 6))
  #  plt.subplot(121)
  #  plt.imshow(img_gc)
  #  plt.axis("off")

  #  plt.subplot(122)
  #  plt.imshow(mask)
  #  plt.axis("off")

   # plt.show()
    cv2.imshow("img",img_gc)
    cv2.waitKey()

if __name__ == '__main__':
    main()