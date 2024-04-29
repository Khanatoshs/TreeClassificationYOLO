import random
import os
import shutil
import configparser


def read_config(filename):
    conf = configparser.ConfigParser()
    conf.read(filename)
    section = 'DATASPLIT'
    src = conf[section].get('sourcefolder')
    dsttrain = conf[section].get('outtrain')
    dstvalid = conf[section].get('outvalid')
    trainpct = conf[section].getint('trainpercentage')
    res_dict = {
       'src': src,
       'dsttrain':dsttrain,
       'dstvalid':dstvalid,
       'trainpct': trainpct
    }
    return res_dict

conf = read_config('config.ini')

src = conf['src']
dst_train = conf['dsttrain']
dst_valid = conf['dstvalid']
tpct = conf['trainpct']

listim = os.listdir(os.path.join(src,'images'))

if not os.path.isdir(dst_train):
    os.mkdir(dst_train)
    os.mkdir(os.path.join(dst_train,'images'))
    os.mkdir(os.path.join(dst_train,'labels'))

if not os.path.isdir(dst_valid):
    os.mkdir(dst_valid)
    os.mkdir(os.path.join(dst_valid,'images'))
    os.mkdir(os.path.join(dst_valid,'labels'))


random.shuffle(listim)

trainlist = listim[:int(tpct/100*len(listim))]
validlist = listim[int(tpct/100*len(listim)):]

if not os.path.isdir(os.path.join(dst_train,'images')):
    os.mkdir(os.path.join(dst_train,'images'))
if not os.path.isdir(os.path.join(dst_train,'labels')):
    os.mkdir(os.path.join(dst_train,'labels'))

if not os.path.isdir(os.path.join(dst_valid,'images')):
    os.mkdir(os.path.join(dst_valid,'images'))
if not os.path.isdir(os.path.join(dst_valid,'labels')):
    os.mkdir(os.path.join(dst_valid,'labels'))


for im in trainlist:
    shutil.copy(os.path.join(src,'images',im),os.path.join(dst_train,'images',im))
    shutil.copy(os.path.join(src,'labels',os.path.splitext(im)[0] + '.txt'),os.path.join(dst_train,'labels'))

for im in validlist:
    shutil.copy(os.path.join(src,'images',im),os.path.join(dst_valid,'images',im))
    shutil.copy(os.path.join(src,'labels',os.path.splitext(im)[0] + '.txt'),os.path.join(dst_valid,'labels'))