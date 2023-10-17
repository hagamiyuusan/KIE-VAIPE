device='cpu'

saliency_ths = 0.65
det_db_box_thresh = 0.6
saliency_weight_path = 'weights/segment/u2netp.pth'
text_detection_dir = 'weights/text_detect'
text_recognition_dir = 'weights/text_rec'
text_extraction_dir = 'weights/kie/gcn.pkl'
rec_char_dict_path = 'weights/text_rec/vi_dict.txt'