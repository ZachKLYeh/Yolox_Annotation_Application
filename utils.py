import numpy as np
import xml.etree.cElementTree as ET
import cv2
import numpy as np
import os
import sys

#detection classes
CLASSES = (
    "person", 
    "car", 
    "motorbike", 
    "bus", 
    "truck", 
    "bike"
)

#color map
_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
    ]
).astype(np.float32).reshape(-1, 3)


#check is the dir exist, if not, then create one
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

#find resouce path, because the reconstructed program will not follow the directory structure
#this is mainly for searching our onnx model
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


#preprocess, including channel transpouse, resize, and int to float transform
def preprocess(img, input_size, swap=(2, 0, 1)):
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)

    return padded_img, r


#nms methods
def nms(boxes, scores, nms_thr):
    """Single class NMS implemented in Numpy."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]

    return keep


def multiclass_nms(boxes, scores, nms_thr, score_thr, class_agnostic=True):
    """Multiclass NMS implemented in Numpy"""
    if class_agnostic:
        nms_method = multiclass_nms_class_agnostic
    else:
        nms_method = multiclass_nms_class_aware

    return nms_method(boxes, scores, nms_thr, score_thr)

def multiclass_nms_class_agnostic(boxes, scores, nms_thr, score_thr):
    """Multiclass NMS implemented in Numpy. Class-agnostic version."""
    cls_inds = scores.argmax(1)
    cls_scores = scores[np.arange(len(cls_inds)), cls_inds]

    valid_score_mask = cls_scores > score_thr

    if valid_score_mask.sum() == 0:
        return None
    valid_scores = cls_scores[valid_score_mask]
    valid_boxes = boxes[valid_score_mask]
    valid_cls_inds = cls_inds[valid_score_mask]
    keep = nms(valid_boxes, valid_scores, nms_thr)

    if keep:
        dets = np.concatenate([valid_boxes[keep], valid_scores[keep, None], valid_cls_inds[keep, None]], 1)

    return dets


#post process for visualizing images
def demo_postprocess(outputs, img_size, p6=False):

    grids = []
    expanded_strides = []

    if not p6:
        strides = [8, 16, 32]
    else:
        strides = [8, 16, 32, 64]

    hsizes = [img_size[0] // stride for stride in strides]
    wsizes = [img_size[1] // stride for stride in strides]

    for hsize, wsize, stride in zip(hsizes, wsizes, strides):
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        expanded_strides.append(np.full((*shape, 1), stride))

    grids = np.concatenate(grids, 1)
    expanded_strides = np.concatenate(expanded_strides, 1)
    outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
    outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

    return outputs


#make xml file after model prediction
def makexml(classes, annotations, filepath, origin_img_shape, input_dir, output_dir):

    boxes, scores, cls_inds = annotations
    boxes, scores, cls_inds = boxes.astype(int), scores.astype(int), cls_inds.astype(int)
    annotation = ET.Element('annotation')
    ET.SubElement(annotation, 'folder').text = str(input_dir)
    ET.SubElement(annotation, 'filename').text = str(os.path.basename(filepath))
    ET.SubElement(annotation, 'path').text = str(os.path.abspath(filepath))
    
    source = ET.SubElement(annotation, 'source')
    ET.SubElement(source, 'database').text = 'Unknown'

    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = str (origin_img_shape[1])
    ET.SubElement(size, 'height').text = str(origin_img_shape[0])
    ET.SubElement(size, 'depth').text = str(origin_img_shape[2])

    ET.SubElement(annotation, 'segmented').text = '0'

    for i in range(len(cls_inds)):
            object = ET.SubElement(annotation, 'object')
            ET.SubElement(object, 'name').text = classes[cls_inds[i]]
            ET.SubElement(object, 'pose').text = 'Unspecified'
            ET.SubElement(object, 'truncated').text = '0'
            ET.SubElement(object, 'difficult').text = '0'

            boxes[i][0] = 0 if boxes[i][0] < 0 else boxes[i][0]
            boxes[i][1] = 0 if boxes[i][1] < 0 else boxes[i][1]
            boxes[i][2] = origin_img_shape[1] if boxes[i][2] > origin_img_shape[1] else boxes[i][2]
            boxes[i][3] = origin_img_shape[0] if boxes[i][0] > origin_img_shape[0] else boxes[i][3]

            bndbox = ET.SubElement(object, 'bndbox')
            ET.SubElement(bndbox, 'xmin').text = str(boxes[i][0])
            ET.SubElement(bndbox, 'ymin').text = str(boxes[i][1])
            ET.SubElement(bndbox, 'xmax').text = str(boxes[i][2])
            ET.SubElement(bndbox, 'ymax').text = str(boxes[i][3])

    tree = ET.ElementTree(annotation)

    if os.path.exists(output_dir):
        mkdir(output_dir)
        xml_file_name = os.path.join(output_dir, os.path.basename(filepath).split('.')[0]+'.xml')
    else:
        xml_file_name = os.path.join(input_dir, os.path.basename(filepath).split('.')[0]+'.xml')

    tree.write(xml_file_name)


#read and xml file, accumelated the classes that it contains
def read_content(xml_file: str, classes, errormsg_record):
    finished = True
    classes = classes
    objects = []  
    list_with_all_boxes = []
    labels = []
    read_content_error_message = errormsg_record
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    filename = root.find('filename').text
    width = int(root.find('size/width').text)
    height = int(root.find('size/height').text)

    for items in root.iter('object'):
        name = items.find("name").text
        
        if name not in classes:
            read_content_error_message += f'{filename} label {name} Not in classes\n'
            finished = False
            break
        
        ymin, xmin, ymax, xmax = None, None, None, None

        ymin = int(items.find("bndbox/ymin").text)
        xmin = int(items.find("bndbox/xmin").text)
        ymax = int(items.find("bndbox/ymax").text)
        xmax = int(items.find("bndbox/xmax").text)
        
        area = (xmax - xmin) * (ymax - ymin)
        labels.append(name)

        if(xmin < 0 or xmin > width or ymin < 0 or ymin > height or
           xmax <= 0 or xmax > width or ymax <= 0 or ymax > height or
           area < 100):
           print(filename, xmin, ymin, xmax, ymax, area)
           read_content_error_message += f'{filename} label size[xmin:{xmin}, ymin:{ymin}, xmax:{xmax}, ymax:{ymax}, area:{area}] is invalid\n'
           finished = False
           break

        labels.append("total")

    return labels, finished, read_content_error_message


#plot bonding boxes on a image, then export
def vis(img, annonation, class_names, image_path, input_dir, output_dir, conf=0.5):

    boxes, scores, cls_ids = annonation

    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

    if os.path.exists(output_dir):
        mkdir(output_dir+'/visualized_images')
        output_path = os.path.join(output_dir,'visualized_images', os.path.basename(image_path))
        cv2.imwrite(output_path, img)
    else:
        mkdir(input_dir+'/visualized_images')
        output_path = os.path.join(input_dir, 'visualized_images', os.path.basename(image_path))
        cv2.imwrite(output_path, img)
