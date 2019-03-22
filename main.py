import operator
import cv2
import numpy as np
from camera import Cam
from flask import Flask, render_template, Response
from platform import system
if False:
    from .camera import Cam

ifwin = str(system()).lower() == 'windows'


app = Flask(__name__)
stream = Cam("192.168.43.54:8080" if ifwin else "127.0.0.1:8080")
stream.start()

hsv_min_white = np.array((51, 0, 180), np.uint8)
hsv_max_white = np.array((171, 134, 255), np.uint8)

hsv_min_red = np.array((0, 152, 73), np.uint8)
hsv_max_red = np.array((206, 232, 162), np.uint8)

hsv_min_green = np.array((54, 77, 56), np.uint8)
hsv_max_green = np.array((86, 255, 255), np.uint8)

hsv_min_yellow = np.array((0, 178, 154), np.uint8)
hsv_max_yellow = np.array((37, 255, 255), np.uint8)

hsv_min_blue = np.array((86, 38, 35), np.uint8)
hsv_max_blue = np.array((190, 204, 164), np.uint8)


@app.route('/')
def index():
    return render_template('index.html')


def check_color(img, min, max, hsv, text):
    thresh = cv2.inRange(hsv, min, max)
    if ifwin:
        contours0, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    else:
        _, contours0, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    found = False
    for cnt in contours0:
        x, y, w, h = cv2.boundingRect(cnt)
        if (200 < h) and (200 < w):
            found = True
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            color = (255, 255, 255)
            if text == "Red":
                color = (255, 0, 0)
            elif text == "Green":
                color = (0, 255, 0)
            elif text == "Yellow":
                color = (255, 255, 0)
            elif text == "Blue":
                color = (0, 0, 255)
            cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
    return found, img


def get_color(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    red, img = check_color(img, hsv_min_red, hsv_max_red, hsv, "Red")
    green, img = check_color(img, hsv_min_green, hsv_max_green, hsv, "Green")
    yellow, img = check_color(img, hsv_min_yellow, hsv_max_yellow, hsv, "Yellow")
    yellow, img = check_color(img, hsv_min_blue, hsv_max_blue, hsv, "Blue")
    return img, red, green, yellow


def rec_num(img):
    global model
    im = img
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, 1, 1, 11, 2)
    if ifwin:
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    else:
        _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    closest = {'x': 0, 'y': 0, 'w': 0, 'h': 0, 'out': -1}
    al = {}
    for cnt in contours:
        [x, y, w, h] = cv2.boundingRect(cnt)
        if (100 < h < 2000) and (
                100 < w < 2000):

            roi = thresh[y:y + h, x:x + w]
            roismall = cv2.resize(roi, (10, 10))
            roismall = roismall.reshape((1, 100))
            roismall = np.float32(roismall)
            retval, results, neigh_resp, dists = model.findNearest(roismall, k=1)
            try:
                al[str(int((results[0][0])))] += 1
            except:
                al[str(int((results[0][0])))] = 1
            ha, wa, _ = im.shape
            ha //= 2
            wa //= 2
            if abs(ha - (y + (h // 2))) < abs(wa - (closest['y'] + (closest['h'] // 2))):
                if abs(wa - (x + (w // 2))) < abs(wa - (closest['x'] + (closest['w'] // 2))):
                    closest['x'] = x
                    closest['y'] = y
                    closest['w'] = w
                    closest['h'] = h
                    closest['out'] = str(int((results[0][0])))
    print(closest['out'])
    print(al)
    try:
        al.pop('-1')
        n = max(al.items(), key=operator.itemgetter(1))[0]
        print("Detected number: " + n)
        return n
    except:
        return closest['out']


def gen():
    while True:
        img = stream.getframe()
        img = cv2.resize(img, (0, 0), fx=0.4, fy=0.4)
        img, red, green, yellow = get_color(img)
        num = rec_num(img)
        w, h, _ = img.shape
        cv2.putText(img, str(num), (15, h // 2 - 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
        _, jpeg = cv2.imencode('.jpg', img)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


model = cv2.ml.KNearest_create()
samples = np.loadtxt('ocr_training.data', np.float32)
responses = np.loadtxt('ocr_responses.data', np.float32)
responses = responses.reshape((responses.size, 1))
model.train(samples, cv2.ml.ROW_SAMPLE, responses)
print("Starting flask...")
app.run(host='0.0.0.0', port=9000, threaded=True)
