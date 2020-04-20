import cv2 as cv
import numpy as np

# a 1 a

img = cv.imread("1.jpg")
nx, ny, nc = img.shape

cv.putText(img, "95101247", (10, nx - 10), cv.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 0), 5)  # add student number

rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # convert to RGB
gray3 = cv.cvtColor(cv.cvtColor(img, cv.COLOR_BGR2GRAY), cv.COLOR_GRAY2BGR)  # convert to GRAY

concat = np.concatenate((img, rgb), axis=1)
concat = np.concatenate((concat, gray3), axis=1)

cv.imshow("a 1 a - BGR-RGB-GRAY", concat)

while True:
    key = cv.waitKey(0)
    if key == ord('s'):  # save and exit
        cv.imwrite("output_a1a.png", concat)
        cv.destroyAllWindows()
        break
    if key == ord('e'):  # exit
        cv.destroyAllWindows()
        break


# a 1 b

img2 = cv.imread("football.jpg")

ball = img2[461:529, 301:375]  # find ball

cv.rectangle(img2, (300, 460), (375, 530), (0, 255, 0))

img2[465:533, 661:735] = ball  # copy ball to other place

cv.imshow("a 1 b", img2)

cv.waitKey(0)
cv.destroyAllWindows()

cv.imwrite("output_a1b.png", img2)


# a 2


def nothing(x):
    pass


img3 = cv.imread("space.jpg")

cv.namedWindow("a 2")

nx, ny, nz = img3.shape
dia = int(np.sqrt(nx ** 2 + ny ** 2) + 1)

back1 = np.zeros((dia, dia, 3), np.uint8)

cv.circle(img3, (100, 100), 10, (0, 255, 0), 3)

back1[620 - 300:320 + nx, 620 - 540:620 + 540] = img3

center = (int(dia / 2), int(dia / 2))

img = np.concatenate((back1, back1), axis=1)

cv.createTrackbar("degree", "a 2", 0, 360, nothing)

py = 100 + 320
px = 100 + 620 - 540

while True:
    cv.imshow("a 2", img)
    deg = cv.getTrackbarPos("degree", "a 2")
    M = cv.getRotationMatrix2D(center, deg, 1)  # rotation matrix
    rotated = cv.warpAffine(back1, M, (dia, dia))  # rotated image
    v = [px, py, 1]
    r = np.dot(M, v)
    cv.circle(rotated, (int(r[0]), int(r[1])), 10, (0, 255, 0), 3)
    img = np.concatenate((back1, rotated), axis=1)
    cv.line(img, (px, py), (dia + int(r[0]), int(r[1])), (0, 0, 255))
    cv.putText(img, "Enter e to exit. :)", (200, 200), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
    k = cv.waitKey(1)
    if k == ord('e'):
        cv.destroyAllWindows()
        break


# a 5


def nothing(x):
    pass


img4 = cv.imread("limbo.png")
img_out = img4.copy()

cv.namedWindow("Erosion")

cv.createTrackbar("Window Size", "Erosion", 1, 40, nothing)

s_element = '1 : MORPH_RECT \n 2 : MORPH_CROSS \n 3 : MORPH_ELLIPSE'

cv.createTrackbar(s_element, "Erosion", 1, 3, nothing)

while True:
    cv.imshow("Erosion", img_out)
    wSize = cv.getTrackbarPos("Window Size", "Erosion")
    sE = cv.getTrackbarPos(s_element, "Erosion")
    if wSize > 0:
        if sE == 1:
            kernel = cv.getStructuringElement(cv.MORPH_RECT, (wSize, wSize))
        if sE == 2:
            kernel = cv.getStructuringElement(cv.MARKER_CROSS, (wSize, wSize))
        if sE == 3:
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (wSize, wSize))
        img_out = cv.erode(img4, kernel)
    k = cv.waitKey(1)
    if k == ord('e'):
        cv.destroyAllWindows()
        break

img_out = img4.copy()

cv.namedWindow("Dilation")
cv.createTrackbar("Window Size", "Dilation", 1, 40, nothing)

s_element = '1 : MORPH_RECT \n 2 : MORPH_CROSS \n 3 : MORPH_ELLIPSE'

cv.createTrackbar(s_element, "Dilation", 1, 3, nothing)

while True:
    cv.imshow("Dilation", img_out)
    wSize = cv.getTrackbarPos("Window Size", "Dilation")
    sE = cv.getTrackbarPos(s_element, "Dilation")
    if wSize > 0:
        if sE == 1:
            kernel = cv.getStructuringElement(cv.MORPH_RECT, (wSize, wSize))
        if sE == 2:
            kernel = cv.getStructuringElement(cv.MARKER_CROSS, (wSize, wSize))
        if sE == 3:
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (wSize, wSize))
        img_out = cv.dilate(img4, kernel)
    k = cv.waitKey(1)
    if k == ord('e'):
        cv.destroyAllWindows()
        break

img_out = img4.copy()

cv.namedWindow("Opening")
cv.createTrackbar("Dilation Window Size", "Opening", 1, 40, nothing)
s_element1 = 'Dilation \n 1 : MORPH_RECT \n 2 : MORPH_CROSS \n 3 : MORPH_ELLIPSE'
cv.createTrackbar(s_element1, "Opening", 1, 3, nothing)
cv.createTrackbar("Erosion Window Size", "Opening", 1, 40, nothing)
s_element2 = 'Erosion \n 1 : MORPH_RECT \n 2 : MORPH_CROSS \n 3 : MORPH_ELLIPSE'
cv.createTrackbar(s_element2, "Opening", 1, 3, nothing)


while True:
    cv.imshow("Opening", img_out)
    img_out = img4.copy()
    wSizeD = cv.getTrackbarPos("Dilation Window Size", "Opening")
    sE1 = cv.getTrackbarPos(s_element1, "Opening")
    if wSizeD > 0:
        if sE1 == 1:
            kernel = cv.getStructuringElement(cv.MORPH_RECT, (wSizeD, wSizeD))
        if sE1 == 2:
            kernel = cv.getStructuringElement(cv.MARKER_CROSS, (wSizeD, wSizeD))
        if sE1 == 3:
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (wSizeD, wSizeD))
        img_out = cv.dilate(img4, kernel)
    wSizeE = cv.getTrackbarPos("Erosion Window Size", "Opening")
    sE2 = cv.getTrackbarPos(s_element2, "Opening")
    if wSizeE > 0:
        if sE2 == 1:
            kernel = cv.getStructuringElement(cv.MORPH_RECT, (wSizeE, wSizeE))
        if sE2 == 2:
            kernel = cv.getStructuringElement(cv.MARKER_CROSS, (wSizeE, wSizeE))
        if sE2 == 3:
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (wSizeE, wSizeE))
        img_out = cv.erode(img_out, kernel)
    k = cv.waitKey(1)
    if k == ord('e'):
        cv.destroyAllWindows()
        break

cv.namedWindow("Closing")
cv.createTrackbar("Erosion Window Size", "Closing", 1, 40, nothing)
s_element1 = 'Erosion \n 1 : MORPH_RECT \n 2 : MORPH_CROSS \n 3 : MORPH_ELLIPSE'
cv.createTrackbar(s_element1, "Closing", 1, 3, nothing)
cv.createTrackbar("Dilation Window Size", "Closing", 1, 40, nothing)
s_element2 = 'Dilation \n 1 : MORPH_RECT \n 2 : MORPH_CROSS \n 3 : MORPH_ELLIPSE'
cv.createTrackbar(s_element2, "Closing", 1, 3, nothing)

img_out = img4.copy()

while True:
    cv.imshow("Closing", img_out)
    img_out = img4.copy()
    wSizeE = cv.getTrackbarPos("Erosion Window Size", "Closing")
    sE1 = cv.getTrackbarPos(s_element1, "Closing")
    if wSizeE > 0:
        if sE1 == 1:
            kernel = cv.getStructuringElement(cv.MORPH_RECT, (wSizeE, wSizeE))
        if sE1 == 2:
            kernel = cv.getStructuringElement(cv.MARKER_CROSS, (wSizeE, wSizeE))
        if sE1 == 3:
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (wSizeE, wSizeE))
        img_out = cv.erode(img4, kernel)
    wSizeD = cv.getTrackbarPos("Dilation Window Size", "Closing")
    sE2 = cv.getTrackbarPos(s_element2, "Closing")
    if wSizeD > 0:
        if sE2 == 1:
            kernel = cv.getStructuringElement(cv.MORPH_RECT, (wSizeD, wSizeD))
        if sE2 == 2:
            kernel = cv.getStructuringElement(cv.MARKER_CROSS, (wSizeD, wSizeD))
        if sE2 == 3:
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (wSizeD, wSizeD))
        img_out = cv.dilate(img_out, kernel)
    k = cv.waitKey(1)
    if k == ord('e'):
        cv.destroyAllWindows()
        break

# a 7

img5 = cv.imread("4.jpg")
img6 = img5.copy()

gray = cv.cvtColor(img5, cv.COLOR_BGR2GRAY)

(thresh1, binaryImage) = cv.threshold(gray, 60, 255, 0)
(thresh2, binaryImage2) = cv.threshold(gray, 25, 255, 0)
pattern = binaryImage[310:334, 80:120]

nx, ny = pattern.shape[::-1]

# method1
res = cv.matchTemplate(binaryImage, pattern, cv.TM_CCOEFF_NORMED)
det = np.where(res >= 0.8)
for dt in zip(*det[::-1]):
    cv.rectangle(img5, dt, (dt[0] + nx, dt[1] + ny), (0, 255, 0), 3)

cv.imshow('a 7 matchTemplate result', img5)
cv.waitKey(0)


# method 2
contours, hierarchy = cv.findContours(binaryImage2, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
old_x = 0
old_y = 0
c = 0

for cnt in contours:
    (x, y, w, h) = cv.boundingRect(cnt)
    if (w > 1.5 * h) and (w < 2 * h) and ((((x - old_x) ** 2 + (y - old_y) ** 2) > 100) or (c == 0)):
        if (y < 350) and (x < 100):
            cv.rectangle(img6, (x, y), (x + w, y + h), (0, 0, 255), 3)
            print("a 7 :")
            print("1st corner :")
            print("x = ", x)
            print("y =", y)
            print("2nd corner :")
            print("x = ", x + w)
            print("y =", y)
            print("3rd corner :")
            print("x = ", x)
            print("y =", y + h)
            print("4th corner :")
            print("x = ", x + w)
            print("y =", y +
                  h)
        else:
            cv.rectangle(img6, (x, y), (x + w, y + h), (0, 255, 0), 3)
        c += 1
        old_y = y
        old_x = x

cv.imshow("a 7 - using contours", img6)
cv.waitKey(0)
cv.destroyAllWindows()

# b 1

video = cv.VideoCapture(0)
saved = cv.VideoWriter('saved.avi', cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (640, 480))
save = False

while True:
    ret, frame = video.read()

    cv.imshow('video', frame)

    if save:
        saved.write(frame)
    k = cv.waitKey(1)
    if k == ord('e'):
        break

    if k == ord('s'):
        save = True

video.release()
saved.release()
cv.destroyAllWindows()

# b 2

video2 = cv.VideoCapture('video.mp4')
fourcc = cv.VideoWriter_fourcc(*'XVID')
ret, frame = video2.read()
height, width, n_channels = frame.shape
out = cv.VideoWriter('b2out.avi', fourcc, 20.0, (width, height), 0)

frames = []
c = 0
while video2.isOpened():
    ret, frame = video2.read()
    c += 1
    if ret:
        if c < 40:
            frames.append(frame)
    else:
        break

video2.release()


background = np.median(frames, axis=0).astype(dtype=np.uint8)
grayBack = cv.cvtColor(background, cv.COLOR_BGR2GRAY)
cv.imshow('background', background)
cv.waitKey(0)

video3 = cv.VideoCapture('video.mp4')
while video3.isOpened():
    ret, frame = video3.read()
    if ret:
        mot = cv.absdiff(cv.cvtColor(frame, cv.COLOR_BGR2GRAY), grayBack)
        mot[np.where(mot < 20)] = 255
        cv.imshow("motions", mot)
        out.write(mot)
        if cv.waitKey(25) & 0xFF == ord('e'):
            break

    else:
        break

out.release()
video3.release()


