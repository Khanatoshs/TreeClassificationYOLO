from ultralytics import YOLO

model50e = 'combined_data_50ex'
model100e = 'combined_data_100ex'
model150e = 'combined_data_150ex'
model200e = 'combined_data_200ex'

resfolder = './testing'
currentmodel = model200e


valfolder = 'results_medium_' + currentmodel

testdta = '/media/smartforest/Storage/Leo/Yonezawa/combined_test.yaml'

modelpath = '/media/smartforest/Storage/Leo/Yonezawa/runs/detect/' + currentmodel + '/weights/best.pt'

model = YOLO(modelpath)

results = model.val(project=resfolder,name=valfolder,save_json=True,device= 0,data = testdta)

print(results.results_dict)