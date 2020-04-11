# Lane Detection - Venkatagiri Ramesh & Rohit Kantharaju
import imutils
import cv2
import numpy as np

def lane_pipeline(frame):
    image = frame
  
    image=cv2.resize(image, (640, 360))

    pts1 =np.float32([[150, 180], [570, 180], [40, 360], [640, 360]])
    pts2 = np.float32([[0, 0], [640, 0], [0, 360], [640, 360]])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    matrix_inv = cv2.getPerspectiveTransform(pts2,pts1)

    pers = cv2.warpPerspective(image, matrix, (640, 360))

    gray = cv2.cvtColor(pers, cv2.COLOR_BGR2GRAY)

    _, threshold = cv2.threshold(gray, 58, 255, cv2.THRESH_BINARY)
   
    median = cv2.medianBlur(threshold,15)


    #left lane processing
    left = median.copy()
    left = left[0:360, 0:270]
    left_img = pers.copy()
    left_img = left_img[0:360,0:270]
    right = median.copy()
    right = right[0:360, 300:640]
    right_img = pers.copy()
    right_img = right_img[0:360,300:640]

    left_cnts = cv2.findContours(left, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    M = cv2.moments(left)
    lcx = int(M["m10"] / M["m00"])
    lcy = int(M["m01"] / M["m00"])
    cv2.circle(left_img, (lcx, lcy), 5, (255, 255, 255), -1)

    cnts = imutils.grab_contours(left_cnts)
    c = max(cnts, key=cv2.contourArea)


    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])
    lbx = extBot[0] + 10
    lby = extBot[1]
    ltx = extTop[0] + 10
    lty = extTop[1]


    lx = [extBot[0]+10,lcx,extTop[0]+10]
    ly = [extBot[1],lcy,extTop[1]]

    l_coeff, res, _, _, _  = np.polyfit(lx,ly,2, full = True)
    lfit = np.poly1d(l_coeff)
    cv2.drawContours(left_img, [c], -1, (0, 255, 255), 2)
    cv2.circle(left_img, (lbx,lby), 6, (255, 0, 0), -1)
    cv2.circle(left_img, (ltx,lty), 6, (0, 255, 0), -1)

    #right lane processing
    right_cnts = cv2.findContours(right, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    M = cv2.moments(right)
    rcx = int(M["m10"] / M["m00"])
    rcy = int(M["m01"] / M["m00"])
    cv2.circle(right_img, (rcx, rcy), 5, (255, 255, 255), -1)

    cnts = imutils.grab_contours(right_cnts)
    c = max(cnts, key=cv2.contourArea)


    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])
    rbx = extBot[0] + 18
    rby = extBot[1]
    rtx = extTop[0] + 18
    rty = extTop[1]


    rx = [extBot[0]+18,rcx,extTop[0]+18]
    ry = [extBot[1],rcy,extTop[1]]

    r_coeff, res, _, _, _  = np.polyfit(rx,ry,2, full = True)
    rfit = np.poly1d(l_coeff)
    cv2.drawContours(right_img, [c], -1, (0, 255, 255), 2)
    cv2.circle(right_img, (rbx,rby), 6, (255, 0, 0), -1)
    cv2.circle(right_img, (rtx,rty), 6, (0, 255, 0), -1)


    poly_pers = pers.copy()
    a3 = np.array( [[[lbx+10,lby],[lcx,lcy],[ltx-5,lty],[rtx+285,rty],[rcx+284,rcy],[rbx+298,rby]]], dtype=np.int32 )
    cv2.fillPoly( pers, a3, 255 )
    alpha = 0.4
    pers= cv2.addWeighted(pers, alpha,poly_pers, 1 - alpha, 0)
    cv2.imshow('Bird View Output',pers)
    pers = cv2.warpPerspective(pers, matrix_inv, (640, 360))
    alpha =0.4
    output_img= cv2.addWeighted(pers, alpha,image, 1 - alpha, 0)

    cv2.imshow('Car View Output',output_img)


    ld = 270 - lbx
    rd = rbx - 70

    offset = rd - ld
    print('Vehicle Offset : ')
    print(offset)

    return left_img,right_img,l_coeff,r_coeff,lfit,rfit,offset

if __name__ == '__main__':

    cap = cv2.VideoCapture('lane_curvefitting.mp4')

    while cap.isOpened():
        _, frame = cap.read()
        l_img,r_img,ll_curve,rl_curve,lfit,rfit,offset =lane_pipeline(frame)
        cv2.imshow('Lane Detection - Left Lane',l_img)
        cv2.imshow('Lane Detection - Right Lane',r_img)
        print(ll_curve)
        print('Right Lane Curve Coefficients :')
        print(rl_curve)
        print('Offset : '+str(offset))
        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    rospy.spin()
