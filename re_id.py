from lightglue.utils import rbd
from lightglue import viz2d
import matplotlib.pyplot as plt
from gui import segment
from lightglue.utils import numpy_image_to_torch, rbd
from similarity import similarity
from gui import segment
import cv2
import numpy as np
import myUtils
import tracker_utils

def getCoordsFrame(frame, initial_bbox, kpts1, kpts0, debug=False):

    x_min, y_min, width, height = initial_bbox

    expansion_factor = 0.10
    new_width = width * (1 + 2 * expansion_factor)
    new_height = height * (1 + 2 * expansion_factor)
    new_x_min = x_min - width * expansion_factor
    new_y_min = y_min - height * expansion_factor

    # Calculate the maximum x and y for the expanded bounding box
    new_x_max = new_x_min + new_width
    new_y_max = new_y_min + new_height

    # Given array of pixel coordinates of shape (376, 2)
    pixel_coordinates = kpts1

    # Find coordinates inside the expanded bounding box
    inside_bbox = []
    inside_indices = []

    for idx, coord in enumerate(pixel_coordinates):
        x, y = coord
        if new_x_min <= x <= new_x_max and new_y_min <= y <= new_y_max:
            inside_bbox.append(coord)
            inside_indices.append(idx)

    inside_bbox = np.array(inside_bbox)
    inside_indices = np.array(inside_indices)
    
    if debug:
        print("Coordinates inside the expanded bounding box:", len(inside_bbox))
        print("Indices of coordinates inside the expanded bounding box:", len(inside_indices))
    
        for i in range(0,len(inside_bbox)):
            print(int(inside_bbox[i][0]),int(inside_bbox[i][1]))

    if len(inside_bbox)>=4:


        # Get the points in ktp0 corresponding to inside_indices
        points_inside_bbox = kpts0[inside_indices]

        # Calculate the bounding box around these points
        x_coords = points_inside_bbox[:, 0]
        y_coords = points_inside_bbox[:, 1]
        x_min = np.min(x_coords)
        y_min = np.min(y_coords)
        x_max = np.max(x_coords)
        y_max = np.max(y_coords)
        
        bounding_box = [int(x_min), int(y_min), int(x_max), int(y_max)]

        #bounding_box=remove_outliers(points_inside_bbox)
        
        if bounding_box[2]==0 or bounding_box[3]==0:
            return None, None
        
        
        if debug:

            print("Points inside the bounding box:", len(points_inside_bbox))
            print("Bounding box around these points:", bounding_box)

            frame_copy=frame.copy()
            cv2.rectangle(frame_copy,(int(bounding_box[0]),int(bounding_box[1])),(int(bounding_box[2]),int(bounding_box[3])),(255, 0, 0), 3)
            cv2.imshow("MATCH",frame_copy)
            cv2.imwrite("MATCH.jpg", frame_copy)
            cv2.waitKey(0)

        return bounding_box, frame[int(bounding_box[1]):int(bounding_box[3]),int(bounding_box[0]):int(bounding_box[2])]
    
    else:

        return None, None


def features_match(frame_init, frame, initial_bbox_scaled, feature_init, scale_x, scale_y, device, extractor, matcher, predictor, vertical=False, debug=False):
    
    if vertical:
        frame = cv2.resize(frame,(480,640),interpolation=cv2.INTER_AREA)
        frame_init = cv2.resize(frame_init,(480,640),interpolation=cv2.INTER_AREA)
    else:
        frame = cv2.resize(frame,(640,480),interpolation=cv2.INTER_AREA)
        frame_init = cv2.resize(frame_init,(640,480),interpolation=cv2.INTER_AREA)
    
    
    image0=numpy_image_to_torch(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))    
    image1=numpy_image_to_torch(cv2.cvtColor(frame_init,cv2.COLOR_BGR2RGB))


    feats0 = extractor.extract(image0.to(device))
    feats1 = extractor.extract(image1.to(device))
    matches01 = matcher({"image0": feats0, "image1": feats1})
    feats0, feats1, matches01 = [
        rbd(x) for x in [feats0, feats1, matches01]
    ]  # remove batch dimension

    kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]

    m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]
        

    if debug:
        print(len(m_kpts0))
        plt.close()
        axes = viz2d.plot_images([image0, image1])
        viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
        viz2d.add_text(0, f'Stop after {matches01["stop"]} layers')
        
        plt.savefig("points_matched.jpg",dpi=100)        
        plt.show()
        plt.close()
        kpc0, kpc1 = viz2d.cm_prune(matches01["prune0"]), viz2d.cm_prune(matches01["prune1"])
        viz2d.plot_images([image0, image1])
        viz2d.plot_keypoints([kpts0, kpts1], colors=[kpc0, kpc1], ps=6)        
        plt.savefig("points_found.jpg",dpi=100)        
        plt.show()

    kpts0=m_kpts0.detach().cpu().numpy()
    kpts1=m_kpts1.detach().cpu().numpy()


    # M, mask = cv2.findHomography(
    #         np.float64([kpts0 for m in matches]).reshape(-1, 1, 2),
    #         np.float64([kpts1 for m in matches]).reshape(-1, 1, 2)
    #     )

    # result = cv2.warpPerspective(frame, M,
    #                                     (frame_init.shape[1], frame_init.shape[0]))


    final_bbox, template_final=getCoordsFrame(frame,initial_bbox_scaled,kpts1,kpts0)    
    
    if debug:
        cv2.imshow("frame",frame)
        cv2.imshow("frame_init",frame_init)
        cv2.waitKey(0)
        cv2.imwrite("image0.jpg",frame)
        cv2.imwrite("image1.jpg",frame_init)           
    
        cv2.destroyAllWindows()
    
    if final_bbox is not None:

        sim=np.round(similarity(feature_init,template_final),2)
        if debug:
            print("SIM", sim)

        if sim > 0.6:
            
            #x1y1wh            
            c=myUtils.center_x1y1x2y2(final_bbox[0],final_bbox[1],final_bbox[2],final_bbox[3])
            seg_image, new_initial_bbox=segment(frame,np.array(c).reshape(1, -1),predictor)    
            seg_image=np.array(seg_image)
            if debug:
                cv2.imshow("seg image",seg_image)
                cv2.imwrite("seg_image.jpg",seg_image)
                cv2.waitKey(0)
            
                cv2.destroyAllWindows()
        else:
            if debug:
                cv2.waitKey(0)

            return False, None

        new_initial_bbox_scaled=[int(new_initial_bbox[0]*(1/scale_x)), int(new_initial_bbox[1]*(1/scale_y)),int(new_initial_bbox[2]*(1/scale_x)), int(new_initial_bbox[3]*(1/scale_y)) ]                
        
        return True, new_initial_bbox_scaled
    
    else:
        if debug:
            cv2.waitKey(0)
        return False, None


def runReID(tracker, count, frame_init, frame, initial_bbox_scaled, feature_init,scale_x, scale_y, device, extractor, matcher, predictor, vertical=False, debug=False):
    
    response, initial_bbox_t=features_match(frame_init, frame,initial_bbox_scaled, feature_init, scale_x, scale_y, device, extractor, matcher, predictor, vertical=vertical)
    if response==False:
        # cv2.putText(frame, "Frame "+str(count)+" Re-ID Failed", (150, 150),
        # cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 3)
        
        return frame, None
        
    else:
        if myUtils.checkBbox(initial_bbox_t, frame.shape[1], frame.shape[0]):
            initial_bbox=initial_bbox_t
            tracker.reset()        

            tracker.initialize(frame, initial_bbox)                          

            frame=tracker_utils.draw_bbox(frame, initial_bbox_t)
            # cv2.putText(frame, "Frame "+str(count)+" Re-ID", (150, 150),
            #     cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 3)
            
        
            return frame, initial_bbox
        else:
            # cv2.putText(frame, "Frame "+str(count)+" Re-ID Failed", (150, 150),
            # cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 3)
            
            return frame, None

    
