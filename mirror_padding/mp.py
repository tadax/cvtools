import cv2
import numpy as np
import dlib

class MirrorPadding:
    def __init__(self, landmarks_dat='./shape_predictor_68_face_landmarks.dat'):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(landmarks_dat)

    def padding(self, img):
        h, w = img.shape[:2]
        r = np.sqrt(h ** 2 + w ** 2) / 2
        res_h = r - h / 2
        res_w = r - w / 2
        
        tmp = np.copy(img)
        for i in range(int(np.ceil(res_h / h * 2)) + 1):
            if i % 4 == 0:
                tmp = np.concatenate((tmp, np.flip(img, 0)), 0)
            elif i % 4 == 1:        
                tmp = np.concatenate((np.flip(img, 0), tmp), 0)
            elif i % 4 == 2:
                tmp = np.concatenate((tmp, img), 0)
            elif i % 4 == 3:
                tmp = np.concatenate((img, tmp), 0)
        if np.ceil(res_h / h * 2 + 1) % 2 == 0:
            img = tmp
        else:
            img = np.roll(tmp, int(h / 2), axis=0)
        stride_h = int((np.ceil(res_h / h * 2) + 1) * h / 2)

        tmp = np.copy(img)
        for i in range(int(np.ceil(res_w / w * 2) + 1)):
            if i % 4 == 0:
                tmp = np.concatenate((tmp, np.flip(img, 1)), 1)
            elif i % 4 == 1:
                tmp = np.concatenate((np.flip(img, 1), tmp), 1)
            elif i % 4 == 2:
                tmp = np.concatenate((tmp, img), 1)
            elif i % 4 == 3:
                tmp = np.concatenate((img, tmp), 1)
        if np.ceil(res_w / w * 2 + 1) % 2 == 0:
            img = tmp
        else:
            img = np.roll(tmp, int(w / 2), axis=1)
        stride_w = int((np.ceil(res_w / w * 2) + 1) * w / 2)

        raw = img[stride_h:stride_h + h, stride_w:stride_w + w, :]
        f = int(max(h, w) / 10)
        img = cv2.blur(img, (f, f))
        img[stride_h:stride_h + h, stride_w:stride_w + w, :] = raw
        
        return img
        

    def get_landmarks(self, img):
        dets = self.detector(img, 1)
        if len(dets) != 1:
            return None
        d = dets[0]
        landmarks = np.float32([(p.x, p.y) for p in self.predictor(img, d).parts()])

        return landmarks


    def effectively_get_landmarks(self, img, productivity):
        threshold = 1000
        h, w = img.shape[:2]

        if max(h, w) <= threshold:
            return self.get_landmarks(img)

        r = threshold / max(h, w)
        new_h = int(h * r)
        new_w = int(w * r)
        new_img = cv2.resize(img, (new_w, new_h))
        landmarks = self.get_landmarks(new_img)

        if landmarks is None:
            if productivity:
                return None
            else:
                return self.get_landmarks(img)
        else:
            return landmarks / r


    def detect(self, img, mirror, productivity=True):
        raw_h, raw_w = img.shape[:2]
        h, w = mirror.shape[:2]
        
        landmarks = self.effectively_get_landmarks(img, productivity)
        if landmarks is None:
            return None

        delta_h = (h - raw_h) / 2
        delta_w = (w - raw_w) / 2
        landmarks[:, 1] += delta_h
        landmarks[:, 0] += delta_w

        e0 = np.mean(landmarks[36:42], axis=0) # right eye
        e1 = np.mean(landmarks[42:48], axis=0) # left eye
        m0 = landmarks[48] # right mouse
        m1 = landmarks[54] # left mouse
        x = e1 - e0
        y = (e0 + e1) / 2 - (m0 + m1) / 2
        c = (e0 + e1) / 2 - 0.1 * y # center
        s = max(4 * np.linalg.norm(x, ord=1), 3.6 * np.linalg.norm(y, ord=1)) # size


        r = -np.arctan(x[1] / x[0])
        R = np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]])
        c = c - [w / 2, h / 2] # translation
        c = np.dot(R, c) + [w / 2, h / 2] # rotation and translation

        left = max(0, int(c[0] - s / 2))
        right = min(w, int(c[0] + s / 2))
        top = max(0, int(c[1] - s / 2))
        bottom = min(h, int(c[1] + s / 2))

        r = -r * 180 / np.pi
        R = cv2.getRotationMatrix2D((mirror.shape[1] / 2, mirror.shape[0] / 2), r, 1)
        mirror = cv2.warpAffine(mirror, R, (mirror.shape[1], mirror.shape[0]))

        mirror = mirror[top:bottom, left:right]

        return mirror

