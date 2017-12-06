import numpy as np
import cv2

def random_flipping(img, p=0.5):
    if np.random.uniform() > p:
        return cv2.flip(img, 1)
    else:
        return img

if __name__ == "__main__":
    img = cv2.imread('test.jpg')
    flipped = random_flipping(np.copy(img), 0.0)
    h, w = img.shape[:2]
    margin = 2
    block1 = np.zeros((h + margin * 2, w + margin * 2, 3), dtype=np.uint8)
    block1[margin:margin + h, margin:margin + w, :] = img
    block2 = np.zeros((h + margin * 2, w + margin * 2, 3), dtype=np.uint8)
    block2[margin:margin + h, margin:margin + w, :] = flipped

    result = np.concatenate((block1, block2), 1)
    cv2.imwrite('results/random_flipping.jpg', result)

