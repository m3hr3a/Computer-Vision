import cv2
import numpy as np
from numpy.linalg import norm

# set video number
video_num = 1
# do you want to show speed on final image?
show_speed = False
# want color box?
color_box = False
# want gradient Lines
show_gd = False
# want number of cars of each line?
show_cn = False
# want to save output?
save_out = False
# want to show background and detection line?
show_bg_dt = False

video = cv2.VideoCapture('Video' + str(video_num) +'.avi')
fps = video.get(cv2.CAP_PROP_FPS)
n_frame = video.get(cv2.CAP_PROP_FRAME_COUNT)
print('length of original video in seconds:')
print(np.round(n_frame/fps, decimals=1))

# load video
frames = []
while True:
    _, frame = video.read()
    if frame is None:
        break
    frames.append(frame)

video.release()

print("Video was Read!")
# estimate background
bg = np.median(frames[:200], axis=0).astype(dtype=np.uint8)

# show background and detection line if wanted
if show_bg_dt:
    if save_out:
        cv2.imwrite("background" + str(video_num) + '.jpg', bg)
    cv2.imshow("background video " + str(video_num), bg)
    cv2.waitKey(5000)
    det_line = bg.copy()
    if video_num == 1:
        det_line = cv2.line(det_line, (0, 300), (800, 300), (255, 0, 0), 2)
    elif video_num == 2:
        det_line = cv2.line(det_line, (0, 260), (800, 25), (255, 0, 0), 2)
    cv2.imshow("detection line " + str(video_num), det_line)
    if save_out:
        cv2.imwrite("detection_line" + str(video_num) + '.jpg', det_line)
    cv2.waitKey(500)
    cv2.destroyAllWindows()

print("background estimated!")

tubes = [f*0+bg for f in frames]
masks = []
start_frame = 0
tube_n = -1
oldp = (0,0)
olds = -100

for frame_number in range(len(frames)):

    frame = frames[frame_number]

    diff = cv2.cvtColor(cv2.absdiff(frame, bg), cv2.COLOR_BGR2GRAY)

    if video_num == 1:
        _, thresh = cv2.threshold(diff, 60, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=2)
        contours, rel = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    elif video_num == 2:
        diff1 = cv2.cvtColor(cv2.absdiff(frame, bg), cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(diff1, 30, 255, cv2.THRESH_BINARY)
        thresh = cv2.medianBlur(thresh, 5)
        dilated = cv2.dilate(thresh, None, iterations=3)
        contours, rel = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for i in range(len(contours)):
        # find bounding rect of object
        (x, y, w, h) = cv2.boundingRect(contours[i])

        if cv2.contourArea(contours[i]) < 200:
            continue

        if video_num == 1:
            cond = y < 300 and y + h > 300 and y+h - 300 < 8 and y+h-300 > 1 and w > 40
        elif video_num == 2:
            d = (np.cross((800, -235), (-x - w, 235 - y - h))) / norm((800, -235))
            move = np.sqrt((x + w - oldp[0]) ** 2 + (y + h - oldp[1]) ** 2)
            mt = abs(frame_number - olds)
            cond = w > 40 and 25 < abs(d) < 30 and d < 0 and (mt > 9.2 or move > 50)
        if cond:
            if video_num == 1:
                (x, y, w, h) = (x-1, y-1, w+2, h+2)
            if rel[0][i][3] == -1:
                tube_n += 1
                if video_num == 2:
                    olds = frame_number
                    oldp = (w + x, y + h)
                old = initBB = ref = (x, y, w, h)
                # detect color start
                detected = frame[y:y + h, x:x + w, :]
                _, diff_d= cv2.threshold(cv2.cvtColor(cv2.absdiff(bg[y:y + h, x:x + w, :], detected), cv2.COLOR_BGR2GRAY),
                                         30, 255, cv2.THRESH_BINARY)
                sig = np.where(diff_d > 0)
                c1 = int(np.mean(detected[sig[0], sig[1], 0]))
                c2 = int(np.mean(detected[sig[0], sig[1], 1]))
                c3 = int(np.mean(detected[sig[0], sig[1], 2]))
                color_t = (c1, c2, c3)
                # detect color end
                # init reverse tracker
                tracker = cv2.TrackerMedianFlow_create()
                tracker.init(frame, initBB)
                d_c = 0
                coords = []
                dists = []
                # reverse tracking
                for t_f in range(frame_number, 0, -1):
                    if initBB is not None:
                        fr_t = frames[t_f]
                        (success, box) = tracker.update(fr_t)

                        if success:
                            (x, y, w, h) = [int(v) for v in box]
                            dist = (x - old[0]) ** 2 + (y - old[1]) ** 2 + (x + w - old[0] - old[2]) ** 2 + (
                                    y + h - old[1] - old[3]) ** 2

                            if dist < 500:
                                dists.append(dist)
                                coords.append((x, y, w, h))
                            start_frame = t_f
                            old = initBB = (x, y, w, h)
                            if dist == 0:
                                d_c += 1
                            else:
                                d_c = 0

                            if d_c == 5:
                                break

                (f_x, f_y, f_w, f_h) = (x, y, w, h)    # first appearance coordination
                # init forward tracker
                tracker = cv2.TrackerMedianFlow_create()
                tracker.init(frame, ref)
                d_c = 0
                for t_f in range(frame_number+1, len(frames), 1):
                    if initBB is not None:
                        fr_t = frames[t_f]
                        (success, box) = tracker.update(fr_t)

                        if success:
                            (x, y, w, h) = [int(v) for v in box]
                            dist = (x - old[0]) ** 2 + (y - old[1]) ** 2 + (x + w - old[0] - old[2]) ** 2 + (
                                      y + h - old[1] - old[3]) ** 2
                            if dist == 0:
                                d_c += 1
                            else:
                                d_c = 0

                            old = initBB = (x, y, w, h)
                            if dist < 500:
                                coords.insert(0, (x, y, w, h))
                                dists.insert(0, dist)
                            if d_c == 5:
                                break

                coords.reverse()
                dists.reverse()
                print("tube " + str(tube_n) + " created")
                # calculate speed
                speed = np.mean(dists)

                start_pos = (f_x, f_y, f_w, f_h)
                masks.append((tube_n, start_frame, coords, start_pos, color_t, speed))

# join tubes
for i in range(len(masks)):
    (tube_n, start_frame, coords, start_pos, color_t, speed) = masks[i]
    (f_x, f_y, f_w, f_h) = start_pos
    for j in range(len(tubes)):
        current = tubes[j]
        cond = True
        for shift in range(int(len(coords)/6)):
            current = tubes[j+shift]
            cond = cond and (current[f_y:f_y + f_h, f_x:f_x + f_w, :] ==
            bg[f_y:f_y + f_h, f_x:f_x + f_w, :]).all()
        if cond:
            for c_n, coord in enumerate(coords):
                (x, y, w, h) = coord
                frame = frames[c_n+start_frame]
                window = frame*0
                window[y:y+h, x:x+w, :] = frame[y:y+h, x:x+w, :]
                time_t = np.round((start_frame+c_n)/fps, decimals=1)
                if color_box:
                    cv2.rectangle(window, (x+2, y+2), (x+w-2, y+h-2), color_t, 4)
                cv2.putText(window, 't:'+str(time_t), (int((2*x+w)/2), int((2*y+h)/2)),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0))
                if show_speed:
                    cv2.putText(window, 's:'+str(np.round(speed,decimals=1)), (int((2 * x + w) / 2),
                                int((2 * y + h) / 2 + 15)),cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0))
                tubes[j+c_n][y:y+h, x:x+w, :] = frame[y:y+h, x:x+w, :]*0
                tubes[j+c_n] = cv2.add(tubes[j+c_n], window)
            break

final_len = 0

if save_out:
    if not (show_speed and color_box):
        saved = cv2.VideoWriter('output'+str(video_num)+'.avi', cv2.VideoWriter_fourcc(*'XVID'), int(fps), (800, 480))
    if show_speed and color_box:
        saved = cv2.VideoWriter('output' + str(video_num) + '._speed_color.avi', cv2.VideoWriter_fourcc(*'XVID'),
                            int(fps), (800, 480))
for i in range(len(tubes)):
    f = tubes[i]
    if (f == bg).all():
        final_len = i
        break
    if save_out:
        saved.write(f)
    cv2.imshow("video synopsis output", f)
    key = cv2.waitKey(40) & 0xff
    #if key == ord("q"):
       # break

if save_out:
    saved.release()
cv2.destroyAllWindows()
print("end of synopsis!")
print("length of synopsised video in seconds:")
print(np.round(final_len/fps, decimals=1))
frames.clear()

# calculate gradient lines and number of cars on each path
if show_gd:
    pathes = []
    for i in range(len(masks)):
        # calculate centroids of each tube
        (tube_n, start_frame, coords, start_pos, color_t, speed) = masks[i]
        (f_x, f_y, f_w, f_h) = start_pos
        path = []
        for c_n, coord in enumerate(coords):
            (x, y, w, h) = coord
            path.append((int((2*x+w)/2), int((2*y+h)/2)))
        pathes.append(path)


    b = bg*0
    i = 0
    X = []
    for p in pathes:
        i = i + 1
        for p2 in p:
            (x,y) = p2
            if video_num == 1:
                th = 220
            elif video_num == 2:
                th = 70
            if y > th:
                cv2.circle(b, p2, 3, (0,255,255), 3)
                X.append([x, y])

    bb = b*0
    lines =[]
    gray = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)
    blured = cv2.blur(thresh, (3,3))
    eroded = cv2.erode(blured, None, iterations=2)
    dilated = cv2.dilate(eroded, None, iterations=5)
    contours, rel = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    linesc = []
    for i in range(len(contours)):
        if cv2.contourArea(contours[i])>1200:
            linesc.append( contours[i])
    for lc in linesc:
        bb = cv2.drawContours(bb, lc, -1, (0,255,0), 3)
        rect = cv2.minAreaRect(lc)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        a1 = int(box[0][0])
        a2 = int(box[0][1])
        a3 = int(box[1][0])
        a4 = int(box[1][1])
        a5 = int(box[2][0])
        a6 = int(box[2][1])
        a7 = int(box[3][0])
        a8 = int(box[3][1])
        w = abs((a1-a3)**2+(a2-a4)**2)
        h = abs((a3-a5)**2+(a4-a6)**2)

        if w > h:
            p1 = (int((a1+a7)/2), int((a2+a8)/2))
            p2 = (int((a3+a5)/2), int((a4+a6)/2))

        else:
            p1 = (int((a1+a3)/2), int((a2+a4)/2))
            p2 = (int((a5+a7)/2), int((a8+a6)/2))

        lines.append((p1,p2))

    n_cars = np.zeros(len(linesc))
    direction = np.zeros(len(linesc))
    for j, l in enumerate(linesc):
        li = lines[j]
        (p1, p2) = li
        (x1, y1) = p1
        (x2, y2) = p2
        if x1 > x2:
            temp = p2
            p2 = p1
            p1 = temp
        (x1, y1) = p1
        (x2, y2) = p2
        disp = []
        for pth in pathes:
            chosen = -1
            d = []
            for i, p in enumerate(pth):
                (x, y) = p
                result = cv2.pointPolygonTest(l, (x, y), False)
                if (result > 0):
                    chosen = j
                    d.append((x1 - x) ** 2 + (y1 - y) ** 2)

            if (chosen != -1):
                ld = len(d)
                s1 = np.sum(d[:int(ld / 2)])
                s2 = np.sum(d[int(ld / 2 + 1):])
                n_cars[chosen] += 1
                if (s1 > s2):
                    direction[j] += 1
                else:
                    direction[j] -= 1

    for i in range(len(direction)):
        if direction[i] > 0:
            direction[i] = -1
        else:
            direction[i] = 1

    bb = bg.copy()
    if show_cn:
        cv2.putText(bb, 'number of cars of each line (in red)', (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
    for j, l in enumerate(lines):
        (p1, p2) = l
        (x1, y1) = p1
        (x2, y2) = p2
        if x1 > x2:
            temp = p2
            p2 = p1
            p1 = temp
        (x1, y1) = p1
        (x2, y2) = p2
        m = (y2 - y1) / (x2 - x1)
        step = abs(int(255/(x2-x1)))+0.5

        for i, x in enumerate(range(x1, x2, 1)):
            y = int(y1 + m * (x - x1))
            if direction[j] > 0:
                c = int(np.max([255 - i * step*2, 0]))
            else:
                c = int(np.min([255, i * step*2]))
            cv2.circle(bb, (x, y), 3, (c, c, c), 10)
        if show_cn:
            cv2.putText(bb, str(n_cars[j]), (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("gradient lines", bb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
