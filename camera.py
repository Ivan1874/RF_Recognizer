import cv2
import numpy as np
import urllib.request as urllib
from threading import Thread

class Cam:
    def __init__(self, host):
        self.hoststr = 'http://' + host + '/video'
        print('Streaming ' + self.hoststr)


    thr = None
    frame = None
    stream = None

    def mainloop(self):
        bytes = b''
        while True:
            bytes += self.stream.read(1024)
            a = bytes.find(b'\xff\xd8')
            b = bytes.find(b'\xff\xd9')
            if a != -1 and b != -1:
                jpg = bytes[a:b + 2]
                bytes = bytes[b + 2:]
                try:
                    self.frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                except Exception as e:
                    print(str(e))

    def start(self):
        self.stream = urllib.urlopen(self.hoststr)
        self.thr = Thread(target=self.mainloop)
        self.thr.start()

    def getframe(self):
        return self.frame
