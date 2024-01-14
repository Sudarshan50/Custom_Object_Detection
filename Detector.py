import cv2, time, os, tensorflow as tf
import numpy as np
from tensorflow.python.keras.utils.data_utils import get_file
np.random.seed(123)
class Detector:
    def __init__(self):
        pass
    
    
    
    def readClasses(self,classFilePath):
        with open(classFilePath ,'r') as f:
            self.classesList = f.read().splitlines()
        
        self.colorList = np.random.uniform(low = 0,high =255,size=(len(self.classesList),3))
        print((len(self.classesList)),len(self.colorList))
        
        
        
    def downloadModel(self,modelURL):
        fileName = os.path.basename(modelURL)
        self.modelName = fileName[:fileName.index('.')]
        self.cacheDir = "./pretrained_Models"
        os.makedirs(self.cacheDir,exist_ok = True)
        get_file(fname = fileName,
        origin = modelURL,cache_dir = self.cacheDir,cache_subdir = "checkpoints",extract = True)
        
        
    def loadModel(self):
        print("Loading model"+self.modelName)
        tf.keras.backend.clear_session()
        self.model = tf.saved_model.load(os.path.join(self.cacheDir,"checkpoints",self.modelName,"saved_model"))
        print("model"+self.modelName+"is loaded successfully")
    
    
    def predictImage(self,imagePath):
        image = cv2.imread(imagePath)
        cv2.imshow("Result",image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        
    

        
    
    
    
