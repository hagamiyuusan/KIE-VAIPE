import cv2
import os 
import time
import torch
import PIL
from PIL import Image
from utils import text_to_json, crop_box, translate_onnx
import onnxruntime
onnxruntime.set_default_logger_severity(3)


import config as cf
from paddleocr import PaddleOCR
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from models.extraction.graph_predict import *
from vietocr.tool.translate import build_model, process_input, translate


class OCR:
    def __init__(self):

        self.detector = PaddleOCR(
            lang='en',
            show_log=False,
            use_space_char='True',
            det_db_box_thresh=0.6,
            drop_score=0.8,
            use_onnx=True,
            det_model_dir='onnx/text_detect.onnx',
            rec_model_dir= 'onnx/rec_onnx.onnx',
            cls_model_dir= 'onnx/cls_onnx.onnx'
        )

        self.configs = Cfg.load_config_from_name('vgg_seq2seq')
        self.configs['cnn']['pretrained']=False
        self.configs['device'] = cf.device
        self.configs['predictor']['beamsearch']=False
        model, vo = build_model(self.configs)
        self.vocab = vo

        cnn_session = onnxruntime.InferenceSession("./onnx/cnn.onnx")
        encoder_session = onnxruntime.InferenceSession("./onnx/encoder.onnx")
        decoder_session = onnxruntime.InferenceSession("./onnx/decoder.onnx")
        self.session = (cnn_session, encoder_session, decoder_session)
        #self.recognizer = Predictor(self.configs)


        node_labels = ['other', 'brandname', 'quantity', 'date', 'usage', 'diagnose', 'generic']
        alphabet = ' "$(),-./0123456789:;ABCDEFGHIJKLMNOPQRSTUVWXYZ_ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝĂĐĨŨƠƯẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼẾỀỂỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰỲỴỶỸ'
        weight = cf.text_extraction_dir
        self.extractor = GRAPH_MODEL(node_labels,alphabet,weight,cf.device)
       
        self.output_logs = {}

    def read(self, img):

        # Text Detection
        bbox = self.detector.ocr(img, rec=False, cls=False)
        bboxes, img_list = crop_box(img, bbox[0])
        data_graph = []
        arr = []

        for img, region in zip(img_list, bboxes):
            img_ocr = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
            img_ocr = process_input(img_ocr, self.configs['dataset']['image_height'], 
                self.configs['dataset']['image_min_width'], self.configs['dataset']['image_max_width'])  
            img_ocr = img_ocr.to(self.configs['device'])
            s = translate_onnx(np.array(img_ocr), self.session)[0].tolist()
            s = self.vocab.decode(s)
            poly = np.array(region).astype(np.int32).reshape((-1))
            if len(s)<2 :
                    continue
            box = np.array(region,np.int32)
            box = box.reshape((-1, 1, 2)) 
            poly = np.array(region).astype(np.int32).reshape((-1))
            # cv2.polylines(image_visualize, [poly.reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=2)
            strResult = '\t'.join([str(p) for p in poly]) +  "\t" + str(s) + "\t" + "other"
            data_graph.append(strResult)
            arr.append(poly)

        t, b, l = self.extractor.predict(data_graph, arr)

        return text_to_json(t, l)

        # return text_to_json(t, l)

    def read_2(self, img):

        # Text Detection
        bbox = self.detector.ocr(img, rec=False, cls=False)
        bboxes, img_list = crop_box(img, bbox[0])
        data_graph = []
        arr = []

        for img, region in zip(img_list, bboxes):
            img_ocr = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
            img_ocr = process_input(img_ocr, self.configs['dataset']['image_height'], 
                self.configs['dataset']['image_min_width'], self.configs['dataset']['image_max_width'])  
            img_ocr = img_ocr.to(self.configs['device'])
            s = translate_onnx(np.array(img_ocr), self.session)[0].tolist()
            s = self.vocab.decode(s)
            poly = np.array(region).astype(np.int32).reshape((-1))
            if len(s)<2 :
                    continue
            box = np.array(region,np.int32)
            box = box.reshape((-1, 1, 2)) 
            poly = np.array(region).astype(np.int32).reshape((-1))
            # cv2.polylines(image_visualize, [poly.reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=2)
            strResult = '\t'.join([str(p) for p in poly]) +  "\t" + str(s) + "\t" + "other"
            data_graph.append(strResult)
            arr.append(poly)

        t, b, l = self.extractor.predict(data_graph, arr)

        return t, b, l




    


