import cv2
cam = cv2.VideoCapture(0)   # 0 -> index of camera
s, img = cam.read()
if s:    # frame captured without any errors
    cv2.namedWindow("cam-test",flags=cv2.WINDOW_AUTOSIZE)
    cv2.imshow("cam-test",img)
    cv2.waitKey(0)
    cv2.imwrite("filename1.jpg",img)
image = cv2.imread("filename1.jpg")
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray Image", gray_image)
cv2.waitKey(0)
output = cv2.cvtColor(gray_image,cv2.COLOR_GRAY2BGR)
cv2.rectangle(output, (300, 100), (500, 400), (255, 138,229), 2)
cv2.line(output,(50,50),(600,50),(223,255,124),2)
cv2.imshow( "Rect Gray Im", output)
cv2.waitKey(0)