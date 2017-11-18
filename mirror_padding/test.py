import numpy as np
import cv2
from mp import MirrorPadding
mp = MirrorPadding()

def test_mirror_padding(path):
    img = cv2.imread(path)
    mirror = mp.padding(img)
    detected = mp.detect(img, mirror)

    thr = 300
    margin = 2

    if detected is None:
        print('Not Detected.')

    else:
        h1, w1 = img.shape[:2]
        r1 = thr / h1
        w1 = int(r1 * w1)
        img = cv2.resize(img, (w1, thr))

        h2, w2 = mirror.shape[:2] 
        r2 = thr / h2
        w2 = int(r2 * w2)
        mirror = cv2.resize(mirror, (w2, thr))

        h3, w3 = detected.shape[:2]
        r3 = thr / h3
        w3 = int(r3 * w3)
        detected = cv2.resize(detected, (w3, thr))

        block1 = np.zeros((thr, margin, 3), dtype=np.uint8)
        block2 = np.zeros((margin, w1 + w2 + w3 + margin * 4, 3), dtype=np.uint8)

        result = np.concatenate((block1, img, block1,  mirror, block1, detected, block1), 1)
        result = np.concatenate((block2, result, block2), 0)

        filename = path.replace('sample', 'result')
        cv2.imwrite(filename, result)

    cv2.destroyAllWindows() 


if __name__ == '__main__':
    for path in ['./results/sample_1.jpg', './results/sample_2.jpg', './results/sample_3.jpg']:
        test_mirror_padding(path)
    
