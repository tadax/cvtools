import numpy as np
import cv2

def random_erasing(img, p=0.5):
    if np.random.uniform() > p:
        return img

    while True:
        s = np.random.uniform() * 0.398 + 0.002 # square to erasing area ratio: [0.02, 0.4]
        r = np.random.uniform() * 9 / 3 # height/width: [1/3, 3]
        img_h, img_w = img.shape[:2]
        area = img_h * img_w * s
        area_w = int(np.sqrt(area / r))
        area_h = int(r * area_w)
        if img_w < area_w or img_h < area_h:
            continue
        x = np.random.randint(0, img_w - area_w + 1)
        y = np.random.randint(0, img_h - area_h + 1)
        img[y:y+area_h, x:x+area_w] = np.random.randint(0, 256)
        #img[y:y+area_h, x:x+area_w, :] = np.random.randint(0, 256, (1, 3))
        return img


if __name__ == "__main__":
    img = cv2.imread('test.jpg')
    augmented = [random_erasing(np.copy(img), p=1.0) for _ in range(20)]
    h, w = img.shape[:2]
    margin = 2
    for i in range(4):
        for j in range(5):
            idx = i * 4 + j
            block = np.zeros((h + margin * 2, w + margin * 2, 3), dtype=np.uint8)
            block[margin:margin + h, margin:margin + w, :] = augmented[idx]
            if j == 0:
                tmp = block
            else:
                tmp = np.concatenate((tmp, block), 1)
        if i == 0:
            result = tmp
        else:
            result = np.concatenate((result, tmp), 0)
    cv2.imwrite('results/random_erasing.jpg', result)

