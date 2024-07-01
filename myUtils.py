from sklearn.covariance import EllipticEnvelope
import cv2
import numpy as np
import os

def checkPath(path):
    
    if not os.path.exists(path):
        os.makedirs(path)
        
    return 

def recordVideo(filename,fps,dim_x,dim_y):
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(filename+'.avi',fourcc, fps, (dim_x,dim_y))
    
    return out

def template_matching(result,template):

    img_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) 

    template=cv2.cvtColor(template,cv2.COLOR_BGR2GRAY)

    # Store width and height of template in w and h 
    h, w = template.shape
    
    # Perform match operations. 
    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED) 
    
    # Specify a threshold 
    #threshold = 0.8
    
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(res)
    print("template conf score ",maxVal)
    (startX, startY) = maxLoc
    endX = startX + template.shape[1]
    endY = startY + template.shape[0]
    
    res=result.copy()
    cv2.rectangle(res, (startX, startY), (endX, endY), (255, 0, 0), 3)    
    cv2.imshow("template match",res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return [startX, startY, endX, endY]


def checkBbox(bbox, width_f, height_f):#xywh

    if bbox[2]<width_f*0.5 and bbox[3]<height_f*0.5:
        return True
    else:
        return False

def increase_shape(bbox, delta=0.1):

    return [bbox[0]-(bbox[2]*delta),bbox[1]-(bbox[3]*delta), bbox[2]+(bbox[2]*delta),bbox[2]+(bbox[3]*delta)]

def remove_outliers(points_inside_bbox):
    
    
    # Fit an ellipse to the data
    elliptic_envelope = EllipticEnvelope(support_fraction=1.0)
    elliptic_envelope.fit(points_inside_bbox)
    print(elliptic_envelope)
    # Get the Mahalanobis distances of each point from the fitted ellipse
    distances = elliptic_envelope.mahalanobis(points_inside_bbox)
    print(distances)

    # Calculate Q1 (25th percentile) and Q3 (75th percentile)
    Q1 = np.percentile(distances, 25)
    Q3 = np.percentile(distances, 75)

    # Calculate IQR (Interquartile Range)
    IQR = Q3 - Q1

    # Determine outlier bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Identify outliers
    outliers = (distances < lower_bound) | (distances > upper_bound)

    print(outliers)

    # Filter out the outliers
    filtered_points = points_inside_bbox[~outliers]
    #print(filtered_points)

    
    for i in range(0,len(filtered_points)):
        print(int(filtered_points[i][0]),int(filtered_points[i][1]))

    filtered_points_array = np.array(filtered_points)

    # Get min and max for x and y
    min_x = np.min(filtered_points_array[:, 0])
    max_x = np.max(filtered_points_array[:, 0])
    min_y = np.min(filtered_points_array[:, 1])
    max_y = np.max(filtered_points_array[:, 1])

    bounding_box = (min_x, min_y, max_x, max_y)

    return bounding_box


def variance_of_laplacian(image):
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
	return cv2.Laplacian(image, cv2.CV_64F).var()
    

def center(x, y, w, h):
    return (x+int(w/2),y+int(h/2))

def center_x1y1x2y2(x1, y1, x2, y2):
    return (x1+int((x2-x1)/2),y1+int((y2-y1)/2))
