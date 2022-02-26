import tensorflow.keras
import numpy as np
import cv2
# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = tensorflow.keras.models.load_model('keras_model.h5')

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Replace this with the path to your image
cam = cv2.VideoCapture(0)

text = ""
text2=""
text3="NAME:"
text4="CALORIES:"
while True:
    # _,img = cv2.imread('right.jpg')
    _,img = cam.read()
    img = cv2.resize(img,(224, 224))
    #turn the image into a numpy array
    image_array = np.asarray(img)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    # print(prediction)
    for i in prediction:
        if i[0] > 0.7:
            text ="apple"
            text2="52 kcal"
        if i[1] > 0.7:
            text ="banana"
            text2="89 kcal"
        if i[2] > 0.7:
            text ="egg"
            text2="155 kcal"
        if i[3] > 0.7:
            text ="brown bread"
            text2="75 kcal"
            
        # print(text)
        img = cv2.resize(img,(500, 500))
        cv2.rectangle(img, (500,500), (3,350), (240,240,240), cv2.FILLED)
        cv2.putText(img,text3,(10,400),cv2.FONT_HERSHEY_COMPLEX_SMALL,2,(0,0,0),1)
        cv2.putText(img,text4,(10,440),cv2.FONT_HERSHEY_COMPLEX_SMALL,2,(0,0,0),1)
        cv2.putText(img,text,(170,400),cv2.FONT_HERSHEY_COMPLEX_SMALL,2,(0,0,0),1)
        cv2.putText(img,text2,(250,440),cv2.FONT_HERSHEY_COMPLEX_SMALL,2,(0,0,0),1)    
    if cv2.waitKey(1) & 0xFF == ord('q'): #This adds a Delay and looks for the key press inorder to break the loop
        cv2.destroyAllWindows()
        break
    cv2.imshow('img',img)
    cv2.waitKey(1)
    cv2.createButton