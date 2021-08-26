import pyzed.sl as sl
import cv2.cv2 as cv
def main():
    zed = sl.Camera()
    InitPara = sl.InitParameters()
    InitPara.camera_fps = 30    
    InitPara.camera_resolution = sl.RESOLUTION.HD720
    err = zed.open(InitPara)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(-1)
    left_img = sl.Mat()
    runtimeParas = sl.RuntimeParameters()    
    while(True):    
        if zed.grab(runtimeParas) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(left_img, view=sl.VIEW.LEFT)
            time_stamp = zed.get_timestamp(sl.TIME_REFERENCE.CURRENT)                        
            img = left_img.get_data()  
            a=left_img.get_width()
            # print('w=',a)
            cv.imshow("img", img)            
            cv.waitKey(5)
if __name__ == "__main__":    main()
