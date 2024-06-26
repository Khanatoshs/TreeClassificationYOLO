import os
from ultralytics import YOLO
from ultralytics import settings

datasrc = 'combined.yaml'
resfolder = './runs'

epochs = 200
name = 'combined_data_'+str(epochs)+'ex'
valfolder = 'results_' + name
imgsize = 700

resultstxt = os.path.join(resfolder,valfolder,'results_' + name + '.txt')

settings.update({'runs_dir':resfolder})
model = YOLO('yolov8n.pt')
results = model.train(data=datasrc,epochs=epochs,imgsz=imgsize,name=name,device=0)
results = model.val(project=resfolder,name=valfolder,save_json=True)


with open(resultstxt,'w+') as res:
    res.write(str(results))
