import numpy as np
import cv2
from affine import Affine
aff = Affine('../shape_predictor_68_face_landmarks.dat')

def test_affine_transformation(path):
    img = cv2.imread(path)
    results, faces = aff.detect_face(img, image_size=img.shape[0])
    assert len(faces) == len(results)
    if len(faces) == 0:
        print('Not Detected.')
    else:
        warped = faces[0]
        print(results)
        top, bottom, left, right = results[0]
        cropped = img[top:bottom, left:right]

        thr = 300
        margin = 2

        h1, w1 = img.shape[:2]
        r1 = thr / h1
        w1 = int(r1 * w1)
        img = cv2.resize(img, (w1, thr))
        block1 = np.zeros((thr + margin * 2, w1 + margin * 2, 3), dtype=np.uint8)
        block1[margin:margin + thr, margin:margin + w1, :] = img

        h2, w2 = cropped.shape[:2] 
        r2 = thr / h2
        w2 = int(r2 * w2)
        cropped = cv2.resize(cropped, (w2, thr))
        block2 = np.zeros((thr + margin * 2, w2 + margin * 2, 3), dtype=np.uint8)
        block2[margin:margin + thr, margin:margin + w2, :] = cropped

        h3, w3 = warped.shape[:2]
        r3 = thr / h3
        w3 = int(r3 * w3)
        warped = cv2.resize(warped, (w3, thr))
        block3 = np.zeros((thr + margin * 2, w3 + margin * 2, 3), dtype=np.uint8)
        block3[margin:margin + thr, margin:margin + w3, :] = warped

        result = np.concatenate((block1, block2, block3), 1)

        filename = path.replace('sample', 'result')
        cv2.imwrite(filename, result)

    cv2.destroyAllWindows() 


if __name__ == '__main__':
    for path in ['./results/sample_1.jpg', './results/sample_2.jpg', './results/sample_3.jpg']:
        test_affine_transformation(path)
    
