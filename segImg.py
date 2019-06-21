# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 22:20:39 2019

@author: fenezema
"""
#IMPORT
from ValidationPreprocess import *
from ModelBuild import *
import sys
#IMPORT

#GLOBAL INIT
model,optimizer = modelBuild()
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.load_weights('saved_weights\\45k_data\\ADAM_0,0001_1000epochs_v3.h5')
labels = {key:chr(key+55) for key in range(10,36)}
#GLOBAL INIT


def getChara(img):
    data_test = []
    pre = ValidationPreprocess()
    imgBin = pre.imageToBinary(redefine={'flag':True,'img':img}, resizeImg = True, sizeImg = 32)
    
    data_test.append(imgBin)
    
    data_test = np.reshape(data_test, (len(data_test), 32, 32, 1))
    
    res = model.predict(data_test)
    pred = return_to_label(res)
    for element in pred:
        if element>9:
            return labels[element]
        else:
            return str(element)
    
def segImg(img,nm_fl):
    the_charas_candidate = {}
    the_charas = []
    try:
        print('masuk try')
        imge = img.copy()
    except:
        return 0,0,0,False
    imggray = cv2.cvtColor(imge,cv2.COLOR_BGR2GRAY)
    h_imggray,w_imggray = imggray.shape
    pre = ValidationPreprocess()
    imgBin = pre.imageToBinary(redefine={'flag':True,'img':imggray})

    kernel = np.ones((3,3),np.uint8)
    erode = cv2.erode(imgBin,kernel,iterations = 1)
    dilate = cv2.dilate(erode,kernel,iterations = 2)
    erode1 = cv2.erode(dilate,kernel,iterations = 1)

    # erode = cv2.erode(imgBin,kernel,iterations = 1)
    # dilate = cv2.dilate(erode,kernel,iterations = 3)
    # erode1 = cv2.erode(dilate,kernel,iterations = 2)
    # dilate1 = cv2.dilate(erode1,kernel,iterations = 1)
    # erode2 = cv2.erode(dilate1,kernel,iterations = 1)
    img1, contours, hierarchy = cv2.findContours(erode1 ,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cou = 0

    for element in contours:
        x,y,w,h = cv2.boundingRect(element)
        if h>w and w/w_imggray > 0.04 and w/w_imggray <=0.15 and h/h_imggray >= 0.29 and h/h_imggray < 0.55:
            print(h,h_imggray,h/h_imggray)
            print(w,w_imggray,w/w_imggray)
            print("masuk if")
            the_charas_candidate[x]=[y,[w,h]]
#            cv2.imwrite('coba/charas/'+nm_fl+'-'+str(cou)+'-'+str(h)+'-'+str(w)+'.jpg',img[y:y+h,x:x+w])
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    
    for ind in sorted(the_charas_candidate.keys()):
        temp = the_charas_candidate[ind]
        w,h = temp[1]
        y = temp[0]
        x = ind
#        cv2.imwrite('coba/charas/'+nm_fl+'-'+str(cou)+'-'+str(h)+'-'+str(w)+'.jpg',imge[y:y+h,x:x+w])
        charnya = getChara( imggray[y:y+h,x:x+w] )
        the_charas.append( charnya )
        cou+=1
    print(the_charas)
    return img,erode1,the_charas,True

def main():
    # theData = open('theData.txt','r')
    while True:
        theData = open('theData.txt','r')
        filename = theData.read()
        # print(filename)
        if filename=='0':
            print('No ROI found')
            result = open('result.txt','w')
            result.write('0')
            result.close()
            theData.close()
            continue
        elif ".jpg" in filename:
            img = cv2.imread(filename)
            newImg,eroded,the_charas,flag = segImg(img,filename)
            if flag == False:
                continue
            charas_nya = ''.join(the_charas)
            result = open('result.txt','w')
            result.write(charas_nya+'.jpg')
            result.close()
            if len(the_charas) == 0:
                print("No Charas Found")
                result = open('result.txt','w')
                result.write('No Charas Found')
                result.close()
            else:
                print(charas_nya)
            theData.close()
        else:
            print('Halted')
            theData.close()
            continue

def main1():
    arg_nya = sys.argv
    img = cv2.imread('toTestedForSync/roi'+str(arg_nya[1])+'.jpg')
    newImg,eroded,the_charas,flag = segImg(img,'roi0.jpg')
    charas_nya = ''.join(the_charas)
    print(charas_nya)
    cv2.imshow('res',newImg)
    cv2.imshow('eroded',eroded)
    cv2.imshow('ori',img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()