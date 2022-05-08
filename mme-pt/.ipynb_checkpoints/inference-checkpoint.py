import numpy as np
import torch
import os
import io
import json
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
import base64

os.system('pip install --upgrade pip')
os.system('pip3 install ipywidgets')


def _image_loads(data):
    """
    Deserializes bytes to stream
    """
    stream = io.BytesIO(base64.b64decode(data))
    return Image.open(stream).convert('RGB')


def input_fn(request_body, CONTENT_TYPE):
    """An input_fn that loads a pickled tensor"""
    print(CONTENT_TYPE)
    if CONTENT_TYPE == 'application/json':
        print("input" + "." * 20)
        print(request_body)
        img_dic = json.loads(request_body)
        target_model = img_dic['target_model']
        img_base = img_dic['img_base'].encode()
        
        img = _image_loads(img_base)
        scaler = transforms.Resize((224, 224))
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        to_tensor = transforms.ToTensor()
        t_img = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))
        print("output" + "." * 20)
        
        infer_data_dic = {'target_model': target_model, 't_img': t_img}
        return infer_data_dic
    else:
        # Handle other content-types here or raise an Exception
        # if the content type is not supported.
        pass


def predict_fn(input_data, model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("start predict" + '.' * 20)
    target_model = input_data['target_model']
    if target_model == 1:
        print("choose model1" + '.' * 20)
        model = model['model1']
        model.eval()
        model.to(device)
        t_img = input_data['t_img'].to(device)
        out = model(t_img)
        print(device)
        print(t_img.is_cuda)
        print("model1 predict done" + '.' * 20)
    else:
        print("choose model2" + '.' * 20)
        model = model['model2']
        model.eval()
        model.to(device)
        t_img = input_data['t_img'].to(device)
        out = model(t_img)
        print(device)
        print(t_img.is_cuda)
        print("model2 predict done" + '.' * 20)
    return out


def output_fn(prediction, content_type):
    print("prediction:{}".format(prediction))
    res = {'result': prediction.tolist()}
    content_type = 'application/json'
    return json.dumps(res), content_type


def model_fn(model_dir):
    print('model_dir:' + str(model_dir))
    print('files:' + str(os.listdir(model_dir)))
    print('dir:' + str(os.getcwd()))

    loaded_model = {
        'model1': torch.load(model_dir + '/img2vec1.pth'),
        'model2': torch.load(model_dir + '/img2vec2.pth')
    }
    return loaded_model
