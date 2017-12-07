import numpy as np
import cv2

def random_blurring(img, kernel_size):
    img = cv2.blur(img, (kernel_size, kernel_size))
    return img

if __name__ == "__main__":
    img = cv2.imread('test.jpg')
    h, w = img.shape[:2]
    s = int(min(h, w) / 20)
    augmented = [random_blurring(np.copy(img), kernel_size=k) for k in np.arange(1, s)]
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
    cv2.imwrite('results/random_blurring.jpg', result)

