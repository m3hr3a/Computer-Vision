import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# A 3

img2 = cv.imread("2.jpg")

img2_resized = cv.resize(img2, (400, 300))    # resize
plt.subplot(421), plt.imshow(cv.cvtColor(img2_resized, cv.COLOR_BGR2RGB)), plt.title('Original image')
plt.xticks([]), plt.yticks([])

# low pass filter

lowPass_kernel = np.ones((5, 5)) / 25  # my low pass kernel
lowPass_img = cv.filter2D(img2_resized, -1, lowPass_kernel)
cv.imwrite('LowPass image.png', lowPass_img)
plt.subplot(422), plt.imshow(cv.cvtColor(lowPass_img, cv.COLOR_BGR2RGB)), plt.title('Low pass image')
plt.xticks([]), plt.yticks([])

# edge detector

edge_kernel = np.ones((3, 3)) * -1
edge_kernel[1, 1] = 8                        # i used Laplacian filter
edges = cv.filter2D(img2_resized, -1, edge_kernel)
cv.imwrite('colored edges.png', edges)
plt.subplot(423), plt.imshow(cv.cvtColor(edges, cv.COLOR_BGR2RGB)), plt.title('colored edges')
plt.xticks([]), plt.yticks([])

gray = cv.cvtColor(img2_resized, cv.COLOR_BGR2GRAY)
gray_edge = cv.filter2D(gray, -1, edge_kernel)
ret, binImage = cv.threshold(gray_edge, 50, 255, cv.THRESH_BINARY)
cv.imwrite('binary edges.png', binImage)
plt.subplot(424), plt.imshow(binImage, 'gray'), plt.title('binary edges')
plt.xticks([]), plt.yticks([])

# high pass using low pass and original
highPass = cv.absdiff(img2_resized, lowPass_img)
cv.imwrite("highPass.png", highPass)
plt.subplot(425), plt.imshow(cv.cvtColor(highPass, cv.COLOR_BGR2RGB)), plt.title('high pass : original - low pass')
plt.xticks([]), plt.yticks([])

# sharpening method 1

sharped = cv.addWeighted(img2_resized, 1, cv.absdiff(img2_resized, lowPass_img), 1.2, 0)
cv.imwrite("sharped1.png", sharped)
plt.subplot(426), plt.imshow(cv.cvtColor(sharped, cv.COLOR_BGR2RGB)), plt.title('sharped image : used original image'
                                                                                'and low pass image')
plt.xticks([]), plt.yticks([])

# sharpening method 2
sharped2 = cv.cvtColor(binImage, cv.COLOR_GRAY2RGB)
sharped2[np.where(sharped2 == 255)] = 1
sharped2 = cv.addWeighted(sharped2 * lowPass_img, 0.6, lowPass_img, 1, 0)
cv.imwrite("sharped2.png", sharped2)
plt.subplot(427), plt.imshow(cv.cvtColor(sharped2, cv.COLOR_BGR2RGB)), plt.title('sharped image : used low pass '
                                                                                 'image and edges')
plt.xticks([]), plt.yticks([])
plt.show()

# A 4

# edges for image 1

img1 = cv.imread("1.jpg")
gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)

sobel_x = cv.Sobel(gray, cv.CV_8U, dx=1, dy=0)
plt.subplot(221), plt.imshow(cv.cvtColor(sobel_x, cv.COLOR_BGR2RGB), cmap='gray'), plt.title('sobel_x edges')
plt.xticks([]), plt.yticks([])

sobel_y = cv.Sobel(gray, cv.CV_8U, dx=0, dy=1)
plt.subplot(222), plt.imshow(cv.cvtColor(sobel_y, cv.COLOR_BGR2RGB), cmap='gray'), plt.title('sobel_y edges')
plt.xticks([]), plt.yticks([])

g = cv.GaussianBlur(gray, (3, 3), 2)
laplacian = cv.Laplacian(g, cv.CV_8U)
laplacian = laplacian / laplacian.max()
laplacian = laplacian.astype('float32')
plt.subplot(223), plt.imshow(cv.cvtColor(laplacian, cv.COLOR_BGR2RGB), cmap='gray'), plt.title('LoG edges')
plt.xticks([]), plt.yticks([])

canny = cv.Canny(gray, 100, 200)
plt.subplot(224), plt.imshow(cv.cvtColor(canny, cv.COLOR_BGR2RGB), cmap='gray'), plt.title('canny edges')
plt.xticks([]), plt.yticks([])

plt.show()

# edges for image 2

img2 = cv.imread("2.jpg")
gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

sobel_x = cv.Sobel(gray, cv.CV_8U, dx=1, dy=0)
plt.subplot(221), plt.imshow(cv.cvtColor(sobel_x, cv.COLOR_BGR2RGB), cmap='gray'), plt.title('sobel_x edges')
plt.xticks([]), plt.yticks([])

sobel_y = cv.Sobel(gray, cv.CV_8U, dx=0, dy=1)
plt.subplot(222), plt.imshow(cv.cvtColor(sobel_y, cv.COLOR_BGR2RGB), cmap='gray'), plt.title('sobel_y edges')
plt.xticks([]), plt.yticks([])

g = cv.GaussianBlur(gray, (3, 3), 2)
laplacian = cv.Laplacian(g, cv.CV_8U)
laplacian = laplacian / laplacian.max()
laplacian = laplacian.astype('float32')
plt.subplot(223), plt.imshow(cv.cvtColor(laplacian, cv.COLOR_BGR2RGB), cmap='gray'), plt.title('LoG edges')
plt.xticks([]), plt.yticks([])

canny = cv.Canny(gray, 100, 200)
plt.subplot(224), plt.imshow(cv.cvtColor(canny, cv.COLOR_BGR2RGB), cmap='gray'), plt.title('canny edges')
plt.xticks([]), plt.yticks([])

plt.show()

# A 6

params = cv.SimpleBlobDetector_Params()

params.minThreshold = 0
params.maxThreshold = 256
params.thresholdStep = 1

params.filterByArea = True
params.minArea = 600

params.filterByCircularity = True
params.minCircularity = 0.5

params.filterByConvexity = True
params.minConvexity = 0.5

params.filterByInertia = True
params.minInertiaRatio = 0.01

im = cv.imread("3.jpg", 0)

detector = cv.SimpleBlobDetector_create(params)
blobs = detector.detect(im)

blobs_out = cv.drawKeypoints(im, blobs, np.array([]), (0, 0, 255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv.imshow("detected blobs", blobs_out)
#cv.imwrite("blobs.png", blobs_out)
cv.waitKey(0)


# b 3

video = cv.VideoCapture('saved.avi')

while video.isOpened():
    ret, frame = video.read()
    if ret:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        sobel_x =  cv.Sobel(gray, cv.CV_8U, dx=1, dy=0)
        cv.imshow("Sobel dx = 1", sobel_x)
        k = cv.waitKey(10)
        if k == ord('q'):
            break
    else:
        break

cv.waitKey(0)

video = cv.VideoCapture('saved.avi')

while video.isOpened():
    ret, frame = video.read()
    if ret:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        sobel_y =  cv.Sobel(gray, cv.CV_8U, dx=0, dy=1)
        cv.imshow("Sobel dy = 1", sobel_y)
        k = cv.waitKey(10)
        if k == ord('q'):
            break
    else:
        break

cv.waitKey(0)

video = cv.VideoCapture('saved.avi')

while video.isOpened():
    ret, frame = video.read()
    if ret:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        canny = cv.Canny(gray, 100, 200)
        cv.imshow("Canny", canny)
        k = cv.waitKey(10)
        if k == ord('q'):
            break
    else:
        break

cv.waitKey(0)

video = cv.VideoCapture('saved.avi')

while video.isOpened():
    ret, frame = video.read()
    if ret:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        kernel_x = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        prewitt_x = cv.filter2D(gray, -1, kernel_x)
        cv.imshow("Prewitt_x", prewitt_x)
        k = cv.waitKey(10)
        if k == ord('q'):
            break
    else:
        break

cv.waitKey(0)

video = cv.VideoCapture('saved.avi')

while video.isOpened():
    ret, frame = video.read()
    if ret:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        kernel_y = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
        prewitt_y = cv.filter2D(gray, -1, kernel_y)
        cv.imshow("Prewitt_y", prewitt_y)
        k = cv.waitKey(10)
        if k == ord('q'):
            break
    else:
        break

cv.waitKey(0)

# Gaussian Sobel dx = 1

video = cv.VideoCapture('saved.avi')

while video.isOpened():
    ret, frame = video.read()
    if ret:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray = cv.GaussianBlur(gray, (3, 3), 2)
        sobel_x_g =  cv.Sobel(gray, cv.CV_8U, dx=1, dy=0)
        cv.imshow("after bluring, Sobel dx = 1", sobel_x_g)
        k = cv.waitKey(10)
        if k == ord('q'):
            break
    else:
        break

cv.waitKey(0)


video = cv.VideoCapture('saved.avi')

while video.isOpened():
    ret, frame = video.read()
    if ret:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray = cv.GaussianBlur(gray, (3, 3), 2)
        sobel_y_g =  cv.Sobel(gray, cv.CV_8U, dx=0, dy=1)
        cv.imshow("after bluring Sobel dy = 1", sobel_y_g)
        k = cv.waitKey(10)
        if k == ord('q'):
            break
    else:
        break

cv.waitKey(0)

video = cv.VideoCapture('saved.avi')

while video.isOpened():
    ret, frame = video.read()
    if ret:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray = cv.GaussianBlur(gray, (3, 3), 2)
        canny_g = cv.Canny(gray, 100, 200)
        cv.imshow("after bluring Canny", canny_g)
        k = cv.waitKey(10)
        if k == ord('q'):
            break
    else:
        break

cv.waitKey(0)

video = cv.VideoCapture('saved.avi')

while video.isOpened():
    ret, frame = video.read()
    if ret:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        kernel_x = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        gray = cv.GaussianBlur(gray, (3, 3), 2)
        prewitt_x_g = cv.filter2D(gray, -1, kernel_x)
        cv.imshow("after bluring Prewitt_x", prewitt_x_g)
        k = cv.waitKey(10)
        if k == ord('q'):
            break
    else:
        break

cv.waitKey(0)

video = cv.VideoCapture('saved.avi')

while video.isOpened():
    ret, frame = video.read()
    if ret:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        kernel_y = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
        gray = cv.GaussianBlur(gray, (3, 3), 2)
        prewitt_y_g = cv.filter2D(gray, -1, kernel_y)
        cv.imshow("after bluring Prewitt_y", prewitt_y_g)
        k = cv.waitKey(10)
        if k == ord('q'):
            break
    else:
        break

cv.waitKey(0)
