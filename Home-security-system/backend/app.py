from flask import Flask, render_template, Response
from camera import VideoCamera


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


def gen(camera):
    while True:
        frame: bytes = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/exec2')
def parse1():
    response_data_collection = VideoCamera().save_to_dataset()
    response_data_collection = "Done with Collecting Data" if response_data_collection else "Do nothing"
    return render_template('index.html', alert=response_data_collection)


@app.route('/training')
def training():
    return render_template('training.html', alert='Not Yet Trained')


if __name__ == '__main__':
    app.run(debug=True)
