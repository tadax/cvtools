# Mirror Padding

## Example

![](samples/result_1.jpg)

![](samples/result_2.jpg)

![](samples/result_3.jpg)


## How to Use

```
$ python
>>> from mp import MirrorPadding
>>> import cv2
>>> img = cv2.imread('data/sample_1.jpg')
>>> mp = MirrorPadding()
>>> mirror = mp.padding(img)
>>> detected = mp.detect(img, mirror)
```

You can download a trained facial shape predictor from:
[shape_predictor_68_face_landmarks.dat](
http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
