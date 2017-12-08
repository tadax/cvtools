import tornado.ioloop
import tornado.web
import cv2
import numpy as np
import io
import os
import base64
from detect import detect

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.render('index.html')

class ApiHandler(tornado.web.RequestHandler):
    def post(self):
        data = self.request.body
        stream = io.BytesIO(data)
        stream.seek(0)
        nparr = np.fromstring(stream.getvalue(), dtype=np.uint8)
        img = cv2.imdecode(nparr, 1)
        detected = detect(img)
        encoded = cv2.imencode('.jpg', detected)[1]
        b64 = base64.b64encode(encoded)
        self.write("data:image/png;base64," + b64.decode('utf-8'))

application = tornado.web.Application([
    (r'/', MainHandler),
    (r'/api', ApiHandler),
    ],
    template_path=os.path.join(os.getcwd(), 'templates'),
)

if __name__ == '__main__':
    application.listen(8888)
    print('Server is up ...')
    tornado.ioloop.IOLoop.instance().start()

