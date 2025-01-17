import cv2

data = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

camera = cv2.VideoCapture(0)

while True:
    rect, img = camera.read()
    convert = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detect = data.detectMultiScale(convert, 1.3, 6)

    for (x1, y1, w1, h1) in detect:
        cv2.rectangle(img, (x1, y1), (x1 + w1, y1 + h1), (255, 0, 0), 5)

    cv2.imshow('image', img)
    t = cv2.waitKey(40) & 0xff
    if t == 27:  # Changed to 27 to break on 'ESC' key
        break

camera.release()
cv2.destroyAllWindows()
