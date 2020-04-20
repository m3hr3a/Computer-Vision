from sklearn.feature_extraction.image import PatchExtractor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from skimage.transform.pyramids import pyramid_gaussian
from skimage import color, feature, data, transform
import skimage
import tensorflow as tf
import numpy as np
import cv2

# Q 2


def calculate_hog_features(img):
    winSize = (64, 128)
    blockSize = (16, 16)
    blockStride = (8, 8)
    cellSize = (8, 8)
    nbins = 9
    derivAperture = 1
    winSigma = 8
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = False
    nlevels = 64
    signedGradients = False

    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins,
                            derivAperture, winSigma, histogramNormType,
                            L2HysThreshold, gammaCorrection, nlevels, signedGradients)

    features = hog.compute(img)
    np.savetxt('hogFeatures', features, delimiter=',')
    print('Features\' shape for (64, 128) image:')
    print(features.shape)


image = cv2.imread('Test_images/test_image1.jpg')
image = cv2.resize(image, (64, 128))
calculate_hog_features(image)


# Q 3

faces = fetch_lfw_people()

positive_patches = faces.images

imgs_to_use = ['camera', 'text', 'coins', 'moon', 'page', 'clock',
               'immunohistochemistry', 'chelsea', 'coffee',
               'hubble_deep_field']

images = [color.rgb2gray(getattr(data, name)())
          for name in imgs_to_use]


def extract_patches(img, n, scale=1.0, patch_size=positive_patches[0].shape):
    extract_patches_size = tuple((scale * np.array(patch_size)).astype(int))
    extractor = PatchExtractor(patch_size=extract_patches_size, max_patches=n, random_state=0)
    patches = extractor.transform(img[np.newaxis])
    if scale != 1:
        patches = np.array([transform.resize(patch, patch_size) for patch in patches])
    return patches


negative_patches = np.vstack([extract_patches(im, 1000, scale) for im in images for scale in [0.5, 1.0, 2.0]])


all_patches = np.concatenate((positive_patches, negative_patches), axis=0)
XTrain = np.array([feature.hog(im) for im in all_patches])
yTrain = np.zeros(XTrain.shape[0])
yTrain[0:positive_patches.shape[0]] = 1


print('size of neagative patches : ')
print(negative_patches.shape)
print('size of positive patches : ')
print(positive_patches.shape)

# Q 4

all_patches = np.concatenate((positive_patches, negative_patches), axis=0)
XTrain = np.array([feature.hog(im) for im in all_patches])
yTrain = np.zeros(XTrain.shape[0])
yTrain[0:positive_patches.shape[0]] = 1

# Q 5

my_grid_search = GridSearchCV(LinearSVC(), {'C': [0.01, 0.02, 0.05, 0.1, 0.2, 1]}, cv=5)
my_grid_search.fit(XTrain, yTrain)
print('Grid Search best accuracy')
print(my_grid_search.best_score_)

X_train, X_test, y_train, y_test = train_test_split(XTrain, yTrain, test_size=0.2, random_state=42)
model1 = my_grid_search.best_estimator_
model1.fit(X_train, y_train)

y_pred = model1.predict(X_test)
print('Chosen model C:')
print(model1.C)
print('Test data accuracy:')
print(accuracy_score(y_test, y_pred))

model = my_grid_search.best_estimator_
model.fit(XTrain, yTrain)

# Q 6


def give_me_boxes(img):
    i = 0
    j = -1
    k = 0
    score = []
    for resized in (pyramid_gaussian(img, downscale=1.2)):
        i = i + 0.5
        j = j + 1
        if resized.shape[0] < 62 or resized.shape[1] < 47:
            break
        for y in range(0, resized.shape[0] - 62, int(32/i)):
            for x in range(0, resized.shape[1] - 47, int(32/i)):
                window = resized[y:y+62, x:x+47]
                label = model.predict(feature.hog(window).reshape((1,1215)))
                p = model.decision_function(feature.hog(window).reshape((1,1215)))
                if label[0] == 1:
                    score.append(p)
                    x2 = int(x * (1.2**j))
                    y2 = int(y * (1.2**j))
                    w = int(47 * (1.2**j))
                    h = int(62 * (1.2**j))
                    if k == 0:
                        k = k + 1
                        boxes = np.array([[y2, x2, y2+h, x2+w]]).T
                    else:
                        boxes = np.concatenate((boxes, np.array([[y2, x2, y2+h, x2+w]]).T)
                                               , axis=1)
    return boxes, score


def my_final_box(boxes, score, img):
    img2 = img.copy()
    img3 = img.copy()
    for box in boxes.T:
        cv2.rectangle(img2, (box[1], box[0]), (box[3], box[2]), (0, 0, 255), 2)
    img1 = img.copy()
    my_boxes = boxes.T
    scores = np.array([s[0] for s in score])
    thresh_score = 1.4
    my_boxes = my_boxes[scores > thresh_score]
    scores = scores[scores > thresh_score]
    for box in my_boxes:
        cv2.rectangle(img3, (box[1], box[0]), (box[3], box[2]), (0, 0, 255), 2)
    selected_indices = tf.image.non_max_suppression(
        my_boxes, scores, 1, 0.8)
    selected_boxes = tf.gather(my_boxes, selected_indices)
    config = tf.ConfigProto(
            device_count={'GPU': 0}
        )
    with tf.Session(config=config) as sess:
        nmsbox = (selected_boxes.eval())
        cv2.rectangle(img, (nmsbox[0, 1], nmsbox[0, 0]), (nmsbox[0, 3], nmsbox[0, 2]), (0, 255, 0), 2)
    heatMap = np.zeros((img.shape[0], img.shape[1])).astype('uint8')
    for (x, y, w, h) in my_boxes:
        heatMap[x:w, y:h] = cv2.add(heatMap[x:w, y:h], 20)
    max_b = np.max(np.max(heatMap))
    heatMap[heatMap < int(max_b / 3)] = 0
    (Y, X) = np.nonzero(heatMap)
    w, h = min(X), min(Y)
    x, y = max(X), max(Y)
    cv2.rectangle(img1, (x, y), (w, h), (0, 255, 0), 2)
    return img2, img3, img, img1


def detect_face(img):
    img1 = skimage.color.rgb2gray(img)
    (boxes, score) = give_me_boxes(img1)
    return my_final_box(boxes, score, img)


def show_and_write(all_boxes, filterd_boxes, nmx, hm, num):
    cv2.imshow('test_image ' + str(num) + ' : all_boxes', all_boxes)
    cv2.waitKey(0)
    cv2.imwrite('test_image' + str(num) + '_all_boxes.jpg', all_boxes)
    cv2.imshow('test_image ' + str(num) + ' : score filtered boxes', filterd_boxes)
    cv2.waitKey(0)
    cv2.imwrite('test_image' + str(num) + '_filtered_boxes.jpg', filterd_boxes)
    cv2.imshow('test_image ' + str(num) + ' : Non max suppression', nmx)
    cv2.waitKey(0)
    cv2.imwrite('test_image' + str(num) + '_NonMaxSuppression.jpg', nmx)
    cv2.imshow('test_image' + str(num) + ' : heatMap', hm)
    cv2.waitKey(0)
    cv2.imwrite('test_image' + str(num) + '_heatMap.jpg', hm)


for i in range(1, 6, 1):
    if i == 4:
        image = cv2.imread('Test_images/test_image' + str(i) + '.png')
    else:
        image = cv2.imread('Test_images/test_image' + str(i) + '.jpg')
    all_boxes, filtered_boxes, nmx, hm = detect_face(image)
    show_and_write(all_boxes, filtered_boxes, nmx, hm, i)
cv2.destroyAllWindows()
