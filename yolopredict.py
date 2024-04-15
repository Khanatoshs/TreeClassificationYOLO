import cv2
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.utils.cv import read_image,visualize_object_predictions
from sahi.predict import get_prediction,get_sliced_prediction,predict
import json
import numpy as np
import logging
import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
#os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "video_codec;h264_cuvid|preset;slow"
logging.basicConfig(filename='./yoloPredict.log',level=logging.DEBUG)

def predict_ortho():

    model50e = 'combined_data_50ex'
    model100e = 'combined_data_100ex'
    model150e = 'combined_data_150ex'
    model200e = 'combined_data_200ex'


    ynz220526 = 'ortho_Yonezawa220526'
    ynz2310_65m = 'ynzw_65m_MPro'
    ynz2310_90m = 'ynzw_90m_MPro'


    ortholist = [ynz220526,ynz2310_65m,ynz2310_90m]
    modellist = [model50e,model150e,model100e,model200e]


    for currentortho in ortholist:
        for currentmodel in modellist:


            modelpath = '/media/smartforest/Storage/Leo/Yonezawa/runs/detect/' + currentmodel + '/weights/best.pt'
            otrhopath = '/media/smartforest/Storage/Leo/Yonezawa/Data/'


            imgpath = otrhopath + currentortho + '.tif'

            predict_dir = '/media/smartforest/Storage/Leo/Yonezawa/predictions'


            detectionModel = AutoDetectionModel.from_pretrained(model_type='yolov8',model_path=modelpath,device=0)


            image = cv2.imread(imgpath,cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            #result = get_prediction(ynz220526,detectionModel)
            result = get_sliced_prediction(image,detectionModel,slice_height=700,slice_width=700,overlap_height_ratio=0.2,overlap_width_ratio=0.2)

            visualize_object_predictions(
                image=np.ascontiguousarray(result.image),
                object_prediction_list=result.object_prediction_list,
                rect_th=2.5,
                text_size=0.8,
                text_th=1.2,
                color=(255,0,0),
                output_dir=predict_dir,
                file_name='boxes_' + currentmodel + '_' +currentortho,
                export_format='png'
            )

            with open(predict_dir+'/predictions_list_' + currentmodel + '_' + currentortho +'.json','w+') as resjson:
                s = json.dumps(result.to_coco_annotations())
                resjson.write(s)

    # prediction_list = result.object_prediction_list
    # print(result.to_coco_predictions(1))
    # resim = cv2.imread(predict_dir + '/prediction_visual.png',cv2.IMREAD_COLOR)
    # cv2.imshow('REsult',resim)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def predict_video():
    model50e = 'combined_data_50ex'
    model100e = 'combined_data_100ex'
    model150e = 'combined_data_150ex'
    model200e = 'combined_data_200ex'

    video = 'DJI_0271'
    extension = '.MP4'
    videopath = '/media/smartforest/Storage/Leo/Yonezawa/Data/videos/' + video + extension

    showout = False

    currentmodel = model150e

    modelpath = '/media/smartforest/Storage/Leo/Yonezawa/runs/detect/' + currentmodel + '/weights/best.pt'

    model = YOLO(modelpath)
    detectionModel = AutoDetectionModel.from_pretrained(model_type='yolov8',model_path=modelpath,device=0)

    cap = cv2.VideoCapture(videopath)
    frame_w,frame_h = int(cap.get(3)),int(cap.get(4))
    fps,fourcc = int(cap.get(5)),cv2.VideoWriter_fourcc(*"mp4v")

    outdir = '/media/smartforest/Storage/Leo/Yonezawa/predictions/'
    outfile = 'videopred_'+video +'_'+ currentmodel + '.mp4'
    writer = cv2.VideoWriter(outdir+outfile,fourcc,fps,(frame_w,frame_h))

    print(cap.isOpened())
    #print(cv2.getBuildInformation())
    while cap.isOpened():
        logging.debug('Opened file: ' + videopath)
        success,frame = cap.read()

        if success:
            #results = model(frame)
            results = get_sliced_prediction(frame,detectionModel,slice_height=700,slice_width=700,overlap_height_ratio=0.2,overlap_width_ratio=0.2)
            prediction_list = results.object_prediction_list

            boxes_list = []
            class_list = []

            for ind,_ in enumerate(prediction_list):
                boxes = (
                    prediction_list[ind].bbox.minx,
                    prediction_list[ind].bbox.miny,
                    prediction_list[ind].bbox.maxx,
                    prediction_list[ind].bbox.maxy
                )
                clss = prediction_list[ind].category.name
                boxes_list.append(boxes)
                class_list.append(clss)

            for box,cls in zip(boxes_list,class_list):
                x1,y1,x2,y2 = box
                cv2.rectangle(frame,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,255),2)
                label = str(cls)
                t_size = cv2.getTextSize(label,0,fontScale=0.6,thickness=1)[0]
                cv2.rectangle(frame,(int(x1),int(y1)-t_size[1]-3),(int(x1)+t_size[0],int(y1)+3),(0,0,255),-1)
                cv2.putText(frame,label,(int(x1),int(y1)-2),0,0.6,[255,255,255],thickness=1,lineType=cv2.LINE_AA)

            if showout:
                cv2.imshow('Inference',frame)

            writer.write(frame)


            #annotated_frame = results[0].plot()

            #cv2.imshow('Inference',annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    writer.release()
    cap.release()
    cv2.destroyAllWindows()

# predict_video()
predict_ortho()