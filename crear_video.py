
import cv2
from picamera2 import Picamera2
import time


def stream_video():
    picam = Picamera2()
    picam.preview_configuration.main.size=(1280, 720)
    picam.preview_configuration.main.format="RGB888"
    picam.preview_configuration.align()
    picam.configure("preview")
    picam.start()
    cont = 0
    sw_grabar = False
    video = []
    while True:
        frame = picam.capture_array()
        if cv2.waitKey(1) & 0xFF == ord('q'):         
            break
        elif cv2.waitKey(1) & 0xFF == ord('s') and not sw_grabar:
            sw_grabar = True
            print("Grabando")
        #if sw_grabar:
            #cv2.imwrite(f'frame{cont}.jpg', frame)
            #cont +=1
        if sw_grabar:
            video.append(frame)
        cv2.imshow("picam", frame)
        #time.sleep(0.01)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    frame_size = (frame.shape[1], frame.shape[0])
    fps = 25
    out = cv2.VideoWriter("video.avi",fourcc,fps,frame_size)
    
    for frame in video:
        out.write(frame)

    out.release()
    cv2.destroyAllWindows()

                                   
if __name__ == "__main__":
    stream_video()
