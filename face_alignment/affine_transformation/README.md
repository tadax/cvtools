# Affine Transformation

## Example

![](results/result_1.jpg)

![](results/result_2.jpg)

![](results/result_3.jpg)


## How to Use

```
$ python
>>> from affine import Affine
>>> import cv2
>>> img = cv2.imread('data/sample_1.jpg')
>>> aff = Affine('../shape_predictor_68_face_landmarks.dat')
>>> results, faces = aff.detect_face(img, image_size)
```

