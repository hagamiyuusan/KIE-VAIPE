import numpy as np 
import cv2
import os
import torch
import PIL
from PIL import Image
import sys 

from .graph_predict import *
from configs import config as cf

class Extractor:
    def __init__(self):
        node_labels = ['other', 'brandname', 'quantity', 'date', 'usage', 'diagnose', 'generic']
        alphabet = ' "$(),-./0123456789:;ABCDEFGHIJKLMNOPQRSTUVWXYZ_ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝĂĐĨŨƠƯẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼẾỀỂỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰỲỴỶỸ'
        weight = cf.kie_weight_path

        self.graph_model = GRAPH_MODEL(node_labels,alphabet,weight,cf.device)

    def get_model(self):
        return self.graph_model
