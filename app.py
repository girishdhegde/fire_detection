import numpy as np
import os
import flask
from flask import Flask,request,jsonify,url_for,render_template
import torch
import cv2
# import shutil
#from werkzeug import secure_filename
from yolo_resnet import yolo
from predict import detect
# import torch
# from torchvision.models import resnet50
torch.hub.download_url_to_file('https://www.dropbox.com/s/a1puv47v6tmrk6j/weights.pt?dl=1', './weight.pt')
# load = './weights.pt'
net = yolo()
net.load_state_dict(torch.load('./weight.pt'))
# resnet = resnet50(pretrained=True)
# net.load_state_dict(load)
net.eval()

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

UPLOAD_FOLDER = "./static"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def home():
    if flask.request.method == "GET":
        return render_template("index.html")
    else:
        # shutil.rmtree('./static')
        # os.mkdir('./static')
        f = request.files["image"]
        path1 = f'./static/{f.filename}'
        path2 = f'./static/outputs/{f.filename}'

        f.save(path1)

        img = cv2.imread(path1)
        out = detect(img, net)
        cv2.imwrite(path2, out)

        return render_template("upload.html", img1=path1, img2=path2)

# @app.route('/prediction', methods=['POST'])
# def predict():
#         f = request.files["image"]
#         path1 = f'./static/{f.filename}'
#         path2 = f'./static/outputs/{f.filename}'
#         f.save(path1)
#         img = cv2.imread(path1)
#         out = detect(img, net)
#         cv2.imwrite(path2, out)
#         return render_template("upload.html", img1=path1, img2=path2)


# @app.route('/prediction',method=['POST'])
# def predict():
#     input_img=request.form.values
#     print(input_img.shape)

#     return render_template("index.html",img=img_path)


if __name__ == "__main__":
    app.run(debug=True)
