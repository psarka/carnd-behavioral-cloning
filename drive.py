import base64
import io

import eventlet.wsgi
import flask
from keras.models import model_from_json
import numpy as np
from PIL import Image
import socketio

# Fix error with Keras and TensorFlow
import tensorflow as tf

from model import preprocess

tf.python.control_flow_ops = tf

sio = socketio.Server()
app = flask.Flask(__name__)
model = None
prev_image_array = None


@sio.on('telemetry')
def telemetry(sid, data):
    _ = sid

    image = Image.open(io.BytesIO(base64.b64decode(data['image'])))
    image_array = preprocess(np.asarray(image))[None, :, :, :]
    steering_angle = float(model.predict(image_array, batch_size=1))

    print('steering angle {}'.format(steering_angle))
    throttle = 0.2
    send_control(steering_angle, throttle)


@sio.on('connect')
def connect(sid, environment):
    _ = environment
    print('connect ', sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit('steer', data={
        'steering_angle': str(steering_angle),
        'throttle': str(throttle),
    }, skip_sid=True)


if __name__ == '__main__':

    with open('model.json', 'r') as f:
        model = model_from_json(f.read())
        print('loaded model.json')

    model.compile('adam', 'mse')
    model.load_weights('model.h5')
    print('loaded weights model.h5')

    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
