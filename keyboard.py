import cv2
import numpy as np

def section_formula(x1,y1,x2,y2,m,n):
    x=(m*x2+n*x1)/(m+n)
    y=(m*y2+n*y1)/(m+n)
    return int(x),int(y)

def add_octaves(pts,list,n):
    [[[x0,y0]],[[x1,y1]],[[x2,y2]],[[x3,y3]]]=pts
    for i in range(n):
        temp=np.zeros((4,1,2), dtype=np.int32)
        x,y=section_formula(x0,y0,x1,y1,i,n-i)
        temp[0][0]=[x,y]
        x,y=section_formula(x0,y0,x1,y1,i+1,n-i-1)
        temp[1][0]=[x,y]
        x,y=section_formula(x3,y3,x2,y2,i+1,n-i-1)
        temp[2][0]=[x,y]
        x,y=section_formula(x3,y3,x2,y2,i,n-i)
        temp[3][0]=[x,y]
        list.append(temp)

def add_white_keys(pts,white):
    [[[x0,y0]],[[x1,y1]],[[x2,y2]],[[x3,y3]]]=pts
    for i in range(7):
        temp=np.zeros((4,1,2), dtype=np.int32)
        x,y=section_formula(x0,y0,x1,y1,i,7-i)
        temp[0][0]=[x,y]
        x,y=section_formula(x0,y0,x1,y1,i+1,7-i-1)
        temp[1][0]=[x,y]
        x,y=section_formula(x3,y3,x2,y2,i+1,7-i-1)
        temp[2][0]=[x,y]
        x,y=section_formula(x3,y3,x2,y2,i,7-i)
        temp[3][0]=[x,y]
        white.append(temp)

def add_black_keys(pts,black):
    [[[x0,y0]],[[x1,y1]],[[x2,y2]],[[x3,y3]]]=pts
    last_x_,last_y_=(x3,y3)
    last_x_up,last_y_up=(x0,y0)
    for i in range(1,7):
        x_,y_=section_formula(x3,y3,x2,y2,i,7-i)
        x_up,y_up=section_formula(x0,y0,x1,y1,i,7-i)
        if i==3:
            last_x_up,last_y_up=x_up,y_up
            last_x_,last_y_=x_,y_
            continue
        temp=np.zeros((4,1,2), dtype=np.int32)
        xy_coords=np.zeros((4,1,2), dtype=np.int32)
        x,y=section_formula(last_x_,last_y_,x_,y_,11,5)
        temp[3][0]=[x,y]
        x,y=section_formula(last_x_up,last_y_up,x_up,y_up,11,5)
        xy_coords[0][0]=[x,y]
        x,y=section_formula(last_x_,last_y_,x_,y_,21,-5)
        temp[2][0]=[x,y]
        x,y=section_formula(last_x_up,last_y_up,x_up,y_up,21,-5)
        xy_coords[1][0]=[x,y]
        x,y=section_formula(temp[2][0][0],temp[2][0][1],xy_coords[1][0][0],xy_coords[1][0][1],5,3)
        temp[1][0]=[x,y]
        x,y=section_formula(temp[3][0][0],temp[3][0][1],xy_coords[0][0][0],xy_coords[0][0][1],5,3)
        temp[0][0]=[x,y]
        last_x_up,last_y_up=x_up,y_up
        last_x_,last_y_=x_,y_
        black.append(temp)

def add_minor_keys(pts,white,black):
    [[[x0,y0]],[[x1,y1]],[[x2,y2]],[[x3,y3]]]=pts
    for i in range(2):
        temp=np.zeros((4,1,2), dtype=np.int32)
        x,y=section_formula(x0,y0,x1,y1,i,2-i)
        temp[0][0]=[x,y]
        x,y=section_formula(x0,y0,x1,y1,i+1,2-i-1)
        temp[1][0]=[x,y]
        x,y=section_formula(x3,y3,x2,y2,i+1,2-i-1)
        temp[2][0]=[x,y]
        x,y=section_formula(x3,y3,x2,y2,i,2-i)
        temp[3][0]=[x,y]
        white.append(temp)
    temp=np.zeros((4,1,2), dtype=np.int32)
    xy_coords=np.zeros((4,1,2), dtype=np.int32)
    x_,y_=section_formula(x3,y3,x2,y2,1,1)
    x_up,y_up=section_formula(x0,y0,x1,y1,1,1)
    x,y=section_formula(x3,y3,x_,y_,11,5)
    temp[3][0]=[x,y]
    x,y=section_formula(x0,y0,x_up,y_up,11,5)
    xy_coords[0][0]=[x,y]
    x,y=section_formula(x3,y3,x_,y_,21,-5)
    temp[2][0]=[x,y]
    x,y=section_formula(x0,y0,x_up,y_up,21,-5)
    xy_coords[1][0]=[x,y]
    x,y=section_formula(temp[2][0][0],temp[2][0][1],xy_coords[1][0][0],xy_coords[1][0][1],5,3)
    temp[1][0]=[x,y]
    x,y=section_formula(temp[3][0][0],temp[3][0][1],xy_coords[0][0][0],xy_coords[0][0][1],5,3)
    temp[0][0]=[x,y]
    black.append(temp)

def get_keyboard_keys(pts,n):
    [[[x0,y0]],[[x1,y1]],[[x2,y2]],[[x3,y3]]]=pts
    white,black=[],[]
    x_up,y_up=section_formula(x0,y0,x1,y1,2,7*n)
    x_,y_=section_formula(x3,y3,x2,y2,2,7*n)
    pts=np.array([[[x0,y0]],[[x_up,y_up]],[[x_,y_]],[[x3,y3]]])
    add_minor_keys(pts,white,black)
    list_of_octaves=[]
    pts=np.array([[[x_up,y_up]],[[x1,y1]],[[x2,y2]],[[x_,y_]]])
    add_octaves(pts,list_of_octaves,n)
    for i in range(n):
        add_white_keys(list_of_octaves[i],white)
        add_black_keys(list_of_octaves[i],black)
    return white,black

def make_keyboard(img,pts,n):
    white,black=get_keyboard_keys(pts,n)
    img=cv2.fillPoly(img,white,(255,255,255))
    img=cv2.polylines(img,white,True,(255,0,255),2)
    img=cv2.fillPoly(img,black,(0,0,0))
    return white,black

def get_slope_and_intercept(x0,y0,x1,y1):
    if x1!=x0:
        m=(y1-y0)/(x1-x0) # there will zero division error
    else:
        m=10000
    b=y1-m*x1
    return m,b

def convert_coordinates_to_lines(white,black):
    white_lines,black_lines=[],[]
    for pts in white:
        temp=np.zeros((4,1,2))
        m,b=get_slope_and_intercept(pts[0][0][0],pts[0][0][1],pts[1][0][0],pts[1][0][1])
        temp[0][0]=[m,b]
        m,b=get_slope_and_intercept(pts[2][0][0],pts[2][0][1],pts[1][0][0],pts[1][0][1])
        temp[1][0]=[m,b]
        m,b=get_slope_and_intercept(pts[2][0][0],pts[2][0][1],pts[3][0][0],pts[3][0][1])
        temp[2][0]=[m,b]
        m,b=get_slope_and_intercept(pts[0][0][0],pts[0][0][1],pts[3][0][0],pts[3][0][1])
        temp[3][0]=[m,b]
        white_lines.append(temp)
    for pts in black:
        temp=np.zeros((4,1,2))
        m,b=get_slope_and_intercept(pts[0][0][0],pts[0][0][1],pts[1][0][0],pts[1][0][1])
        temp[0][0]=[m,b]
        m,b=get_slope_and_intercept(pts[2][0][0],pts[2][0][1],pts[1][0][0],pts[1][0][1])
        temp[1][0]=[m,b]
        m,b=get_slope_and_intercept(pts[2][0][0],pts[2][0][1],pts[3][0][0],pts[3][0][1])
        temp[2][0]=[m,b]
        m,b=get_slope_and_intercept(pts[0][0][0],pts[0][0][1],pts[3][0][0],pts[3][0][1])
        temp[3][0]=[m,b]
        black_lines.append(temp)
    
    return white_lines,black_lines

def point_and_line(m,b,x,y):
    y_=x*m+b
    if y<=y_:
        return True
    else:
        return False

def check_key(x,y,white_lines,black_lines):
    for i,key in enumerate(black_lines):
        [[[m0,b0]],[[m1,b1]],[[m2,b2]],[[m3,b3]]]=key
        if point_and_line(m0,b0,x,y) ^ point_and_line(m2,b2,x,y) and point_and_line(m1,b1,x,y) ^ point_and_line(m3,b3,x,y):
            print("Black ",i)
            return ("Black",i)
    for i,key in enumerate(white_lines):
        [[[m0,b0]],[[m1,b1]],[[m2,b2]],[[m3,b3]]]=key
        if point_and_line(m0,b0,x,y) ^ point_and_line(m2,b2,x,y) and point_and_line(m1,b1,x,y) ^ point_and_line(m3,b3,x,y):
            print("White ",i)
            return ("White",i)
    return None

def call_back(event,x,y,flags,param):
    if event == cv2.EVENT_RBUTTONDOWN:
        print(x,y)
        check_key(x,y,white_lines,black_lines)

if __name__=='__main__':
    shape=(800,600)
    img=np.zeros((shape[0],shape[1],3), dtype=np.uint8)
    (x0,y0)=(200,400)
    (x1,y1)=(400,400)
    (x2,y2)=(500,500)
    (x3,y3)=(100,500)
    n=3
    pts=np.array([[[x0,y0]],[[x1,y1]],[[x2,y2]],[[x3,y3]]])
    white,black=make_keyboard(img,pts,n)
    white_lines,black_lines=convert_coordinates_to_lines(white,black)
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Image",img)
    cv2.resizeWindow('Image', 400, 400)
    cv2.setMouseCallback('Image',call_back)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
