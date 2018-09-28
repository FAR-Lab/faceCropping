import cv2
import sys
import os
import os.path

folderPath = sys.argv[1]
fileEnding = ".jpg"
imagePath = '014600_image10.jpg'
cascPath =  'haarcascade_frontalface_default.xml'
FacePictureSize =224


# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

os.mkdir(folderPath+'output/')

for dirpath, dirnames, filenames in os.walk(folderPath):
    for filename in [f for f in filenames if f.endswith(fileEnding)]:
        print (os.path.join(dirpath, filename))
        image = cv2.imread(os.path.join(dirpath, filename))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Detect faces in the image
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
            #flags = cv2.CV_HAAR_SCALE_IMAGE
        )
        ImageWidth = len(image[0])
        ImageHeight = len(image)
        for (x, y, w, h) in faces:
            xn = int((x+(h/2.0))-(FacePictureSize/2.0))
            yn = int((y+(w/2.0))-(FacePictureSize/2.0))
            if(xn<0):
                xn=0
            if(xn+FacePictureSize>ImageWidth):
                xn = ImageWidth-FacePictureSize
            if (yn<0):
                yn=0
            if(yn+FacePictureSize>ImageHeight):
                yn = ImageHeight-FacePictureSize
                #print("From {0} , {1} to {2}, {3}".format(xn,yn,xn+FacePictureSize,yn+FacePictureSize))
            newimage = image[yn:yn+FacePictureSize,xn:xn+FacePictureSize]
            cv2.imwrite(os.path.join((folderPath+'output/'),filename),newimage)
