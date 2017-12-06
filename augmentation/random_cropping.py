import numpy as np
import cv2

def random_cropping(img, s=0.5):
    h, w = img.shape[:2]
    r = np.random.uniform(0, s)
    v1 = np.random.randint(0, int(r * h) + 1)
    v2 = np.random.randint(0, int(r * w) + 1)
    img = img[v1:v1+int((1 - r) * h), v2:v2+int((1 - r) * w), :]
    img = cv2.resize(img, (w, h))

    return img

if __name__ == "__main__":
    img = cv2.imread('test.jpg')
    augmented = [random_cropping(np.copy(img)) for _ in range(20)]
    #np.arange(0.0, 0.6, 0.03)]
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
    cv2.imwrite('results/random_cropping.jpg', result)

