import argparse
import os
import platform
import cv2
import yaml
from yaml.loader import SafeLoader
import sys
from pathlib import Path
import torch
import cv2
import numpy as np
import os



FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative



from ultralytics.utils.plotting import Annotator, colors, save_one_box
from common import DetectMultiBackend
from general import (masks2segments, process_mask, process_mask_native, increment_path, 
                                  Profile, non_max_suppression, print_args, scale_boxes, scale_segments, IMG_FORMATS, LoadImages)



def run(
    weights=ROOT / '',  # model.pt path(s)
    source=ROOT / '',  # file/dir/URL/glob/screen/0(webcam)
    data=ROOT / '',  # dataset.yaml path
    imgsz=(640, 640),  # inference size (height, width)
    conf_thres=0.25,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=1000,  # maximum detections per image
    device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img=False,  # show results
    save_txt=False,  # save results to *.txt
    save_conf=False,  # save confidences in --save-txt labels
    save_crop=False,  # save cropped prediction boxes
    nosave=False,  # do not save images/videos
    classes=None,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    augment=False,  # augmented inference
    visualize=False,  # visualize features
    update=False,  # update all models
    project=ROOT / '2_Predictions/runs',  # save results to project/name
    name='exp',  # save results to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    line_thickness=3,  # bounding box thickness (pixels)
    hide_labels=False,  # hide labels
    hide_conf=False,  # hide confidences
    half=False,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    vid_stride=1,  # video frame-rate stride
    retina_masks=False):



    source = str(source)
    img = cv2.imread(source)
    # ----------------------------------------------
    cv2.imshow('source image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # ----------------------------------------------
    print(img.shape)# length and height and RGB - original's copy image
    row, col, d = img.shape
    print("height pixel :",row)
    print("length pixel :",col)
    print("RGB :",d)
    # ----------------------------------------------
    save_img = not nosave and not source.endswith('.txt')  # save inference images

    


    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run



    # Load model
    device = torch.device('cpu')
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = [640, 640]



    # Dataloader
    bs = 1  # batch_size
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs



    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]: 
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0   
            # ---------------------------------------------- 
            # print("printing im shape before permute",im.shape)
            # print("printing im type",type(im))   
            im_np = im.permute(1, 2, 0).numpy()
            # print("printing im shape after permute",im_np.shape)
            # print("printing im type",type(im_np))   
            # ----------------------------------------------
            cv2.imshow('Image', im_np)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            # ----------------------------------------------
            print(im_np.shape)# length and height and RGB - original's copy image
            row, col, d = im_np.shape
            print("height pixel :",row)
            print("length pixel :",col)
            print("RGB :",d)
            # ---------------------------------------------- 
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred, proto = model(im, augment=augment, visualize=visualize)[:2]


        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det, nm=32)



        with open(ROOT / '2_Predictions/data.yaml',mode='r') as f:
            data_yaml = yaml.load(f,Loader=SafeLoader)
    
        class_names = data_yaml['names']
        class_count = {name: 0 for name in class_names}


        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if False:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                if retina_masks:
                    # scale bbox first the crop masks
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size
                    masks = process_mask_native(proto[i], det[:, 6:], det[:, :4], im0.shape[:2])  # HWC
                else:
                    masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size

                # Segments
                if save_txt:
                    segments = [
                        scale_segments(im0.shape if retina_masks else im.shape[2:], x, im0.shape, normalize=True)
                        for x in reversed(masks2segments(masks))]

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    class_name = names[int(c)]
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    class_count[class_name] = class_count.get(class_name, 0) + n.item()
                # print("printing s",s)

                # Mask plotting
                annotator.masks(
                    masks,
                    colors=[colors(x, True) for x in det[:, 5]],
                    im_gpu=torch.as_tensor(im0, dtype=torch.float16).to(device).permute(2, 0, 1).flip(0).contiguous() /
                    255 if retina_masks else im[i])

                # draw conf score, class name, bounding box
                for j, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
                    if save_txt:  # Write to file
                        seg = segments[j].reshape(-1)  # (n,2) to (n*2)
                        line = (cls, *seg, conf) if save_conf else (cls, *seg)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))

                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)


            # show results (image with detections)
            cv2.imshow('predicted image', im0)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            # ----------------------------------------------
            print(im0.shape)# length and height and RGB - original's copy image
            row, col, d = im0.shape
            print("height pixel :",row)
            print("length pixel :",col)
            print("RGB :",d)

                
        
        print(class_count)




def parse_opt(onnx_model, test_img, data_yaml):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / onnx_model, help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / test_img, help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / data_yaml, help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / '2_Predictions/runs', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    parser.add_argument('--retina-masks', action='store_true', help='whether to plot masks in native resolution')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    run(**vars(opt))



