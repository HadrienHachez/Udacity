import sys
if(len(sys.argv) != 2):
    print("wrong args")
    exit()

import numpy as np
import cv2
import utils
import socketio
import eventlet
import base64
from io import BytesIO
from PIL import Image
from keras.models import load_model

# Socket.IO server
sio = socketio.Server()

# event from simulator
@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        # retrive data
        speed = float(data["speed"])
        steering_angle = float(data["steering_angle"])
        throttle = float(data["throttle"])
        image = Image.open(BytesIO(base64.b64decode(data["image"])))

        try:
            image = np.asarray(image)      
            
            # apply the preprocessing
            if mode==1:
                image = utils.processImg(image)
            else:
                image = utils.preprocess(image)

            image = np.array([image])       # the model expects 4D array

            # Get Prediction
            steering_angle = float(model.predict(image, batch_size=1))
            # calculate throttle
            throttle = 1.0 - speed/25

            send(steering_angle, throttle)
        except Exception as e:
            print(e)
    else:
        sio.emit('manual', data={}, skip_sid=True)

# event fired when simulator connect
@sio.on('connect')
def connect(sid, environ):
    send(0, 0)

# simulator send command
def send(steer, throttle):
    sio.emit("steer", data={'steering_angle': str(steer), 'throttle': str(throttle)}, skip_sid=True)


# wrap with a WSGI application
app = socketio.WSGIApp(sio)

# simulator will connect to localhost:4567
if __name__ == '__main__':
    mode=int(sys.argv[1])
    if mode == 2:
        model = load_model("./models/model_vgg.h5")
    elif mode == 1:
        model = load_model("./models/model_basic.h5")
    else:
        print("wrong args")
        exit()
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)

# simulator will connect to localhost:4567
if __name__ == '__main__':
    model = load_model("./models/model_30-20000-40-relu.h5")
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
