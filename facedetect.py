import cv2
import uuid
import boto3
s3 = boto3.client('s3')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
count = 0
vidcap = cv2.VideoCapture(0)
success, img = vidcap.read()
success = True

while success:
    success, img = vidcap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        roi_color = img[y-70:y+h+70, x-50:x+w+50]
        if roi_color.size:
            image_string = cv2.imencode('.jpg', roi_color)[1].tostring()
            s3.put_object(Bucket="silencio-persons", Key=uuid.uuid4().hex + ".jpg", Body=image_string)
            print('Read a new frame: ', count)
            count = count + 1
    cv2.imshow('img', img)
    k = cv2.waitKey(1000) & 0xff
    if k == 27:
        break

vidcap.release()
cv2.destroyAllWindows()
