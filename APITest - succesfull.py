
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import skimage.io as io
import pylab
pylab.rcParams['figure.figsize'] = (10.0, 8.0)



annType = ['segm','bbox','keypoints']
annType = annType[2]      #specify type here
prefix = 'person_keypoints' if annType=='keypoints' else 'instances'
print ('Running demo for *%s* results.' %(annType))


#initialize COCO ground truth api
dataDir='../'
dataType='val2014'
#annFile = '%s/annotations/%s_%s.json'%(dataDir,prefix,dataType)
annFile = 'person_keypoints_val2017.json'
cocoGt=COCO(annFile)


#initialize COCO detections api
resFile='pratice_kyepoint.json'
#resFile = resFile%(dataDir, prefix, dataType, annType)
cocoDt=cocoGt.loadRes(resFile)


###############Finding AP for all images################
import json
dts = json.load(open(resFile,'r'))
imgIds = [imid['image_id'] for imid in dts]
imgIds = sorted(list(set(imgIds)))
del dts


################Finding AP for range of images#############

#imgIds=sorted(cocoGt.getImgIds())
#imgIds=imgIds[0:200]
#imgId = imgIds[np.random.randint(200)]


# running evaluation
cocoEval = COCOeval(cocoGt,cocoDt,annType)
cocoEval.params.imgIds  = imgIds
cocoEval.params.catIds=[1] # 1 stands for the 'person' class, can use to find AP for specific class.
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()



