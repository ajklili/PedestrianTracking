import cv2
import sys
import blobs
import numpy as np

fix_back = True
area = [1160, 300, 100, 400]  # [x, y, w, h], interested area
alpha = 0.5


def inside(point, area):
    if area[0] < point[0] < area[0] + area[2] and area[1] < point[1] < area[1] + area[3]:
        return True
    else:
        return False


def show_video(video_file):
    tracker = blobs.BlobTracker()
    inners = []
    max_id = 0
    area_4 = [area[0], area[1], area[0] + area[2], area[1] + area[3]]

    video = video_file
    print(video, 'fixed background')

    c = cv2.VideoCapture(video)
    _, f = c.read()
    c_zero = np.float32(f)
    c.set(0, 000.0)
    width = int(c.get(3))
    height = int(c.get(4))
    fps = c.get(5)
    fourcc = c.get(6)
    frames = c.get(7)
    print(fourcc, fps, width, height, frames)
    print()
    # whole_area = [0, 0, width, height]

    if fix_back:
        for_er = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        for_di = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 23))
        for_cl = np.ones((19, 31), np.uint8)
    else:
        for_er = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        for_di = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 25))
        for_cl = np.ones((10, 10), np.uint8)

    trails = np.zeros((height, width, 3)).astype(np.uint8)
    current_frame = 0

    ''' imshow window '''
    cv2.namedWindow('output', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('output', 192 * 5, 108 * 5)

    back = cv2.imread('demo_0.png')
    back = cv2.cvtColor(back, cv2.COLOR_BGR2GRAY)
    back = cv2.equalizeHist(back)

    while True:
        _, f = c.read()
        current_frame = c.get(1)

        if fix_back:
            gray_image = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            gray_image_eq = cv2.equalizeHist(gray_image)
            im_diff = cv2.absdiff(gray_image_eq, back)
            im_pre = cv2.medianBlur(im_diff, 3)
            thresh, im_bw = cv2.threshold(im_pre, 55, 255, cv2.THRESH_BINARY)
        else:
            cv2.accumulateWeighted(f, c_zero, 0.25)
            im_zero = c_zero.astype(np.uint8)
            im_diff_1 = cv2.absdiff(f, im_zero)
            im_diff = cv2.cvtColor(im_diff_1, cv2.COLOR_BGR2GRAY)
            im_pre = cv2.medianBlur(im_diff, 3)
            thresh, im_bw = cv2.threshold(im_pre, 10, 255, cv2.THRESH_BINARY)

        im_er = cv2.erode(im_bw, for_er)
        im_dl = cv2.dilate(im_er, for_di)
        im_p = cv2.morphologyEx(im_dl, cv2.MORPH_CLOSE, for_cl)

        im2, contours, hierarchy = cv2.findContours(im_p, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        my_blobs = []
        for cnt in contours:
            try:
                x, y, w, h = cv2.boundingRect(cnt)
                # cv2.rectangle(f, (x, y), (x + w, y + h), (255, 0, 0), 2)
                moments = cv2.moments(cnt)
                x = int(moments['m10'] / moments['m00'])
                y = int(moments['m01'] / moments['m00'])
                my_blobs.append((x, y))
            except Exception:
                print("Bad Rect")

        if len(my_blobs) > 0:
            tracker.track_blobs(my_blobs, area_4, current_frame)
            for v in tracker.virtual_blobs:
                size = 10
                cv2.rectangle(f, (int(v.x), int(v.y)), (int(v.x + size), int(v.y + size)), v.color, size)

        for id in tracker.traces:
            max_id = max(max_id, id)
            ox = None
            oy = None
            if len(tracker.traces[id]) > 2:
                for pos in tracker.traces[id][-3:]:
                    x = int(pos[0])
                    y = int(pos[1])
                    if inside((x, y), area):
                        if id not in inners:
                            inners.append(id)
                    if ox and oy:
                        sx = int(0.8 * ox + 0.2 * x)
                        sy = int(0.8 * oy + 0.2 * y)
                        oy = sy
                        ox = sx
                    else:
                        ox, oy = x, y

        cv2.rectangle(f, (area[0], area[1]), (area[0] + area[2], area[1] + area[3]), (255, 255, 0), 3)
        cv2.add(f, trails, f)
        # cv2.drawContours(f, contours, -1, (0, 255, 0), 1)

        if int(current_frame) % 25 == 0:
            print("Video percentage:", int((current_frame / frames) * 100), 
                ", Estimated people in screen:", len(contours), 
                ", Estimated total people walked into the area:", int(len(inners) * alpha))
        cv2.imshow('output', f)
        cv2.waitKey(1)

    cv2.destroyAllWindows()
    c.release()


if __name__ == "__main__":
    show_video(sys.argv[1])
