import numpy as np
import cv2
import utils
import socketio
import eventlet
import base64
from io import BytesIO
from PIL import Image
from keras.models import load_model

INPUT_SHAPE = (160, 320, 3) # height, width, channels

MAX_SPEED = 15
MIN_SPEED = 5
speed_limit = MAX_SPEED

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
            image = utils.preprocess(image) # apply the preprocessing
            image = np.array([image])       # the model expects 4D array

            # Get Prediction
            steering_angle = float(model.predict(image, batch_size=1))

            # set speed limits (downhill, ..)
            global speed_limit
            if speed > speed_limit:
                speed_limit = MIN_SPEED 
            else:
                speed_limit = MAX_SPEED
            
            # calculate throttle
            throttle = 1.0 - steering_angle**2 - (speed/speed_limit)**2

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
    model = load_model("./models/model_30-20000-40-relu.h5")
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
