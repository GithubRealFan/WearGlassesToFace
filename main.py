import cv2
import dlib
import numpy as np
import math

inputImage = "image/glass.png"
glassesImage = "image/8.jpg"

# Load the face and landmark detector models
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Load the glasses image
glasses_img = cv2.imread(inputImage, cv2.IMREAD_UNCHANGED)

# Load the input face image
img = cv2.imread(glassesImage)

# Convert to grayscale for faster processing
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect the face landmarks
rects = face_detector(gray)
landmarks = landmark_predictor(gray, rects[0])

# Extract the eye and nose landmarks
leftEye = (landmarks.part(36).x, landmarks.part(36).y)
rightEye = (landmarks.part(45).x, landmarks.part(45).y)
nose = (landmarks.part(28).x, landmarks.part(28).y)
noset = (landmarks.part(27).x, landmarks.part(27).y)

d1 = math.sqrt((leftEye[0] - noset[0])**2 + (leftEye[1] - noset[1])**2)
d2 = math.sqrt((rightEye[0] - noset[0])**2 + (rightEye[1] - noset[1])**2)

# Estimate the pose of the face
left_eye_landmarks = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
right_eye_landmarks = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]
left_eye_center = (int(sum([p[0] for p in left_eye_landmarks]) / len(left_eye_landmarks)), int(sum([p[1] for p in left_eye_landmarks]) / len(left_eye_landmarks)))
right_eye_center = (int(sum([p[0] for p in right_eye_landmarks]) / len(right_eye_landmarks)), int(sum([p[1] for p in right_eye_landmarks]) / len(right_eye_landmarks)))

dY = rightEye[1] - leftEye[1]
dX = rightEye[0] - leftEye[0]
dY2 = right_eye_center[1] - left_eye_center[1]
dX2 = right_eye_center[0] - left_eye_center[0]
angle = np.arctan2(dY2, dX2)

w = glasses_img.shape[1]
h = glasses_img.shape[0]

if w > 1000:
    ratio = 1000. / w
    glasses_img = cv2.resize(glasses_img, (int(w * ratio), int (h * ratio)))

w = int(w * ratio)
h = int(h * ratio)

scale = math.sqrt(dX ** 2 + dY ** 2) / glasses_img.shape[1] * 1.5
eyesCenter = ((leftEye[0] + rightEye[0]) // 2, (leftEye[1] + rightEye[1]) // 2)

cosa = np.cos(angle)
sina = np.sin(angle)

h2 = h * 0.65

m11 = cosa * scale
m12 = -sina * scale
m13 = (h2 * sina - w * cosa / 2) * scale + nose[0]

m21 = sina * scale
m22 = cosa * scale
m23 = (-h2 * cosa - w * sina / 2) * scale + nose[1]

whiteX = nose[0]
whiteY = nose[1]

w = glasses_img.shape[1]
h = glasses_img.shape[0]

iw = img.shape[1]
ih = img.shape[0]

tem = np.zeros((ih, iw, 4), dtype=np.int32)
cnt = np.zeros((ih, iw, 4), dtype=np.int32)

dp1 = (d1 + d2) / max(d1, d2) / 2
dp2 = dp1

if d1 < d2:
    dp2 = 1
else:
    dp1 = 1

rh = (dp1 + dp2) * w / 4 - w / 2

for x in range(w):
    xr = x / w
    xx = x * (xr * dp2 + (1-xr) * dp1)
    xx = xx - rh
    for y in range(h):
        px = xx * m11 + y * m12 + m13
        py = xx * m21 + y * m22 + m23
        npx = int(px)
        npy = int(py)
        if 0 <= npx and npx < iw and 0 <= npy and npy < ih:
            for c in range(4):
                tem[npy][npx][c] = tem[npy][npx][c] + int(glasses_img[y][x][c])
                cnt[npy][npx][c] = cnt[npy][npx][c] + 1

for x in range(iw):
    for y in range(ih):
        if cnt[y][x][0] > 0:
            for c in range(4):
                tem[y][x][c] = tem[y][x][c] // cnt[y][x][c]

for x in range(iw):
    for y in range(ih):
        if tem[y][x][3] > 0:
            for c in range(3):
                img[y][x][c] = min(255, int((img[y][x][c] * (255 - tem[y][x][3]) + tem[y][x][c] * tem[y][x][3]) / 255))

# Show the output image
cv2.imwrite("result.jpg", img)
cv2.imshow("Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

