import configparser
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
#import to manipulate shape files for gps coordinates
import fiona
import rasterio
import cv2
import logging

logging.basicConfig(filename='imageprocess.log',encoding='utf-8',filemode='w' ,level=logging.DEBUG)
fiona.log.setLevel(logging.INFO)
rasterio.log.setLevel(logging.INFO)

def read_config(filename):
    conf = configparser.ConfigParser()
    conf.read(filename)
    section = 'IMAGEPROCESS'
    shapelist = conf[section].get('shapelist').split(',')
    tifflist = conf[section].get('tifflist').split(',')
    imsize = conf[section].getint('imsize')
    savepath = conf[section].get('savepath')
    categories = conf[section].get('category').split(',')
    treesize = conf[section].getint('treesize')
    res_dict = {
       'shapefile': shapelist,
       'tifflist':tifflist,
       'imsize':imsize,
       'savepath':savepath,
       'treesize':treesize,
       'categories':categories
    }
    return res_dict

# Let's define a couple of Functions to read images

#Read color image
def read_Color_Image(path):
    retVal = cv2.imread(path,cv2.IMREAD_COLOR)
    if retVal is None: raise Exception("Reading Color image, something went wrong with the file name "+str(path))
    return retVal

#Read binary image
def read_Binary_Mask(path):
    retVal = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    if retVal is None: raise Exception("Reading GRAYSCALE image, something went wrong with the file name "+str(path))
    retVal[retVal<=50] = 0
    retVal[retVal>50] = 255
    return retVal


def point_in_area(image,point,coordB,coordE):
   px,py = image.index(*point['coordinates'])
   if px <0 or py <0:
      return False
   if px < coordB[0] or px > coordE[0] or py < coordB[1] or py > coordE[1]:
      return False
   return True

def polygon_in_area(image,polygon,coordB,coordE):
    
    if type(polygon['coordinates'][0]) == list :
      poly = polygon['coordinates'][0]
      if type(poly[0]) == list :
        poly = polygon['coordinates'][0][0]
    else:
      poly = polygon['coordinates']
    logging.debug("POLY: " + str(poly))
    for cor in poly:
        #logging.debug(cor)
        px,py = image.index(*cor)
        if px <0 or py <0:
            return False
        if px < coordB[0] or px > coordE[0] or py < coordB[1] or py > coordE[1]:
           return False
    return True

def polygon_process(image,polygon,coordB,coordE):
  pxmin,pxmax,pymin,pymax = 0,0,0,0
  if type(polygon['coordinates'][0]) == list :
    poly = polygon['coordinates'][0]
    if type(poly[0]) == list :
      poly = polygon['coordinates'][0][0]
  else:
    poly = polygon['coordinates']
  inside = False
  for cor in poly:
        #logging.debug(cor)
        px,py = image.index(*cor)
        if (px > coordB[0] and  py > coordB[1]) and (px < coordE[0] and py < coordE[1]):
           inside = True
           break
  if inside:
    pxmin,pxmax,pymin,pymax = get_tree_coords(image,polygon)
    pxc = pxmin + (pxmax - pxmin //2)
    pyc = pymin + (pymax - pymin //2)
    if pxc < coordB[0] or pxc > coordE[0] or pyc < coordB[1] or pyc > coordE[1]:
      inside = False
  return inside,pxmin,pxmax,pymin,pymax



def polygon_inside(image,polygon):
  ''' (image: OBJ, polygon: dict) -> bool
  checks if the polygon coordinates are inside the current image
  image: rasterio opened image
  polygon: a polygon from the list of polygons read from the shapefile

  Returns True if all teh coordinates of the polygon are inside the image

  '''
  for cor in polygon['coordinates']:
    px,py = image.index(*cor)
    if px <0 or py <0:
      return False
  return True


def map_shape(shape):
    id = shape['id']
    coordinates =  shape['geometry']['coordinates']
    resdict = {
        'Fid':id,
        'coordinates':coordinates,
        'type':'Point'
    }
    resdict.update(shape['properties'])
    return resdict

def get_list_coords(shape_file_list,categories):
  '''(shape_file_list:list[str] ) -> list[dict]

      Reads the shapefiles provided and outputs a list with the id of the polygon and the coordinates of the polygon 
      output format: [{id:str, coordinates:list}] 
      coordinates: [gpscoordinate: tuple] -> each element is a tuple of gps coordinates
  
  '''
  shapes = []
  
  for ci,shape_file in enumerate(shape_file_list):
    with fiona.open(shape_file,'r') as src_shp:
      geoType = str(src_shp.schema['geometry'])
      if geoType == 'Point':
        shapes_aux = list(map(lambda i:map_shape(i),src_shp))
      else:
        shapes_aux = list(
            #map the elements read in the shape file to a list of dicts where the first element is the id and the second 
            map(lambda i: 
                {'type':'Polygon','category':categories[ci],'coordinates': i['geometry']['coordinates'][0] if len(i['geometry']['coordinates']) <=1 else i['geometry']['coordinates']
                },src_shp))  
            #map(lambda i: 
          #   {'id': i['properties']['MERGE_SRC'] + '_' + str(i['properties']['id']),
            #   'coordinates': i['geometry']['coordinates'][0] if len(i['geometry']['coordinates']) <=1 else i['geometry']['coordinates']
            #  },poly))  
      
      shapes.extend(shapes_aux)
  return shapes

def get_tree_coords_point(image,point,treesize):
  pyc,pxc = image.index(*point['coordinates'])
  pxmin = pxc - (treesize//2)
  pymin = pyc - (treesize//2)
  pxmax = pxc + (treesize//2)
  pymax = pyc + (treesize//2)
  return pxmin,pxmax,pymin,pymax

def get_tree_coords(image,polygon):
    
  #start pixels at min and max so at the end a square for the whole polygon can be cut
  if type(polygon['coordinates'][0]) == list :
    poly = polygon['coordinates'][0]
    if type(poly[0]) == list :
      poly = polygon['coordinates'][0][0]
  else:
    poly = polygon['coordinates']
  pydef,pxdef = image.index(*poly[0])

  pxmin = pxdef
  pxmax = pxdef
  pymin = pydef
  pymax = pydef
    
  for cor in poly:
    py,px = image.index(*cor)
    if px >= pxmax:
      pxmax = px
    elif px <= pxmin:
      pxmin = px
    if py >= pymax:
      pymax = py
    elif py <= pymin:
      pymin = py
  return pxmin,pxmax,pymin,pymax


def get_tree_image(rgb_image,polygon):
  '''(rgb_image:str, polygon:dict) -> cv2image

     Finds the tree with the coordinates indicated in the polygon element and
     and returns a image cropped from the rgb image input 
  
  '''
  with rasterio.open(rgb_image) as dsm:
    #start pixels at min and max so at the end a square for the whole polygon can be cut
    pxmin,pxmax,pymin,pymax = get_tree_coords(dsm,polygon)
  #load rgb image
  fullim = read_Color_Image(rgb_image)
  #crop image with min and max values
  crop = fullim[pymin:pymax, pxmin:pxmax]
  crop = crop[:,:,::-1]
  return crop

def coord_to_yolo(pxmin,pxmax,pymin,pymax,imh,imw,pxleft = 0,pyup = 0):
  pxwidth = pxmax - pxmin
  pxheight = pymax - pymin
  pxc = pxmin + (pxwidth//2)
  pyc = pymin + (pxheight//2)
  yolx = (pxc - pxleft)/imw
  yoly = (pyc - pyup)/imh
  yolw = pxwidth/imw
  yolh = pxheight/imh
  return {'x': yolx,
          'y': yoly,
          'width': yolw,
          'height': yolh}

def save_yolo_file(yolodata,filename):
  with open(filename,'w') as yolotxt:
    for data in yolodata:
      yolotxt.write(str(data['cat']) + ' ' + str(data['x']) + ' ' + str(data['y']) + ' ' + str(data['width']) + ' ' + str(data['height']) + '\n')

def split_tiff(imtiff,size,location,polys,rastertiff,name,treesize,category):
    logging.debug(range(0,imtiff.shape[0],size))
    logging.debug(range(0,imtiff.shape[1],size))
    for r in range(0,imtiff.shape[0],size):
        for c in range(0,imtiff.shape[1],size):
            cutIm = imtiff[r:r+size, c:c+size,:]
            h,w,ch = cutIm.shape
            listYolo = []
            for p in polys:
                inArea = False
                if p['type'] == 'Point':
                  if point_in_area(rastertiff,p,(r,c),(r+size,c+size)):
                     pxmin,pxmax,pymin,pymax = get_tree_coords_point(rastertiff,p,treesize)
                     inArea = True
                else:
                  # try:
                  # inArea,pxmin,pxmax,pymin,pymax = polygon_process(rastertiff,p,(r,c),(r+size,c+size))
                  if polygon_in_area(rastertiff,p,(r,c),(r+size,c+size)):
                      pxmin,pxmax,pymin,pymax = get_tree_coords(rastertiff,p)
                      inArea = True
                  # except TypeError as e:
                  #    logging.debug(p)
                  #    logging.error(e)
                  #    exit()
                if inArea:
                  dictYolo = coord_to_yolo(pxmin,pxmax,pymin,pymax,h,w,c,r)
                  dictYolo['cat'] = p['category']
                  listYolo.append(dictYolo)
            if len(listYolo) > 0:
                save_yolo_file(listYolo,os.path.join(location,'labels',name + f"_{r}_{c}.txt"))
                cv2.imwrite(os.path.join(location,'images',name + f"_{r}_{c}.png"),imtiff[r:r+size, c:c+size,:])

conf = read_config('config.ini')
pathShapes = conf['shapefile']
listMosaics = conf['tifflist']
imsize = conf['imsize']
savepath = conf['savepath']
treesize = conf['treesize']
categories = conf['categories']
shapes = get_list_coords(pathShapes,categories)

for mosaic in listMosaics:
   rastertiff = rasterio.open(mosaic)
   image = read_Color_Image(mosaic)
   logging.debug(image.shape)
   basename = os.path.splitext(os.path.basename(mosaic))[0] 
   split_tiff(image,imsize,savepath,shapes,rastertiff,'im' + basename,treesize,categories)


