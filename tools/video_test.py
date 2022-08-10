import cv2

cap = cv2.VideoCapture("test_video.mp4")

while True:
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    print(ret)
    if cv2.waitKey(1) and 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()