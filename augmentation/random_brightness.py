import numpy as np
import cv2

def random_brightness(img, gamma=None):
    if gamma is None:
        gamma = np.random.rand() * 6 + 0.1
    gf = [[255 * pow(i/255, 1/gamma)] for i in range(256)]
    table = np.reshape(gf, (256, -1))
    img = cv2.LUT(img, table)
    return img

if __name__ == "__main__":
    img = cv2.imread('test.jpg')
    augmented = [random_brightness(np.copy(img), gamma=g) for g in np.arange(0.1, 6.1, 0.2)]
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
    cv2.imwrite('results/random_brightness.jpg', result)

