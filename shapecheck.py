from fiona.crs import to_string
import fiona
import rasterio
import configparser
import geopandas
import os


def read_config(filename):
    conf = configparser.ConfigParser()
    conf.read(filename)
    section = 'DATACHECK'
    
    shapelist = conf[section].get('shapelist').split(',')
    tiff = conf[section].get('tiff')
    rprjpath = conf[section].get('reprojectedpath')
    res_dict = {
       'shapelist': shapelist,
       'tiff':tiff,
       'rprjpath':rprjpath
    }
    return res_dict


conf = read_config('config.ini')
shapelist = conf['shapelist']
rprjpath = conf['rprjpath']
tiff = conf['tiff']

tiffcrs = None
with rasterio.open(tiff) as raster:
    print('Ortho CRS:',raster.crs)
    tiffcrs = str(raster.crs)

newfile = None
changed = False
for shape in shapelist:
    changed = False
    shapegeo = geopandas.read_file(shape)
    print('Shapefile CRS:', str(shapegeo.crs))
    if str(shapegeo.crs) != tiffcrs:
        print('Orthomosaic CRS is different from shapefile CRS, reprojecting.......')
        newcrs = shapegeo.to_crs(tiffcrs)
        newfile = os.path.join(rprjpath,'reprojected_' + os.path.basename(shape))  
        newcrs.to_file(newfile)
        print('Created file:',newfile)
        changed = True
    else:
        print('CRS is the same for shapefile and Orthomosaic no changes needed')
    
#FINAL CHECK JUST IN CASE
if changed:
    with fiona.open(newfile) as newshp:
        if str(newshp.crs) != tiffcrs:
            print('Somehow CRS is still not the same')
            print('Shapefile CRS')

