import warnings
warnings.filterwarnings('ignore')
import argparse
import os
sep=os.sep
import cv2
import numpy as np
from typing import List
import sys
sys.path.insert(0,'LightGlue')
sys.path.insert(0,'FEARTracker')
from lightglue import LightGlue, SuperPoint
import torch
import myUtils
import tracker_utils
import re_id
import gui
import gui_video
from lightglue import LightGlue, SuperPoint
import copy
from similarity import extract_features, similarity
import time
from tqdm import tqdm
from sam import loadPredictor

def main(
    video_path: str,
    initial_bbox: List[int],        
    config_path: str = "model_training/config",
    config_name: str = "fear_tracker",
    weights_path: str = "evaluate/checkpoints/FEAR-XS-NoEmbs.ckpt",
):
    vertical=False
    max_points=500
    num_blurred=0
 
    output_path="outputs/proc_"+video_path[video_path.rfind(sep)+1:]
    myUtils.checkPath(output_path[:output_path.rfind(sep)])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'

    extractor = SuperPoint(max_num_keypoints=max_points).eval().cuda()  # load the extractor
    matcher = LightGlue(features='superpoint').eval().cuda()  # load the matcher

    predictor=loadPredictor()

    tracker = tracker_utils.get_tracker(config_path=config_path, config_name=config_name, weights_path=weights_path)
    
    initial_bbox = np.array(initial_bbox)
    
    
    cap = cv2.VideoCapture(video_path)

    # Check if camera opened successfully
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
    
    total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
    count=0   
    skipped=0
    start_skipped=time.time()

    noReID=True
    for i in tqdm(range(0,total_frame_count)):
        if cap.isOpened()==False:
            break
        
        ret, frame = cap.read() 

        if ret == True:
            count=count+1
            frame_c=frame.copy()   
                                    
            if count==1:
                if frame.shape[0]>frame.shape[1]:
                    video=myUtils.recordVideo(output_path,30,1080,1920)
                    vertical=True
                    
                    scale_y=640/1920
                    scale_x=480/1080
                else:
                    video=myUtils.recordVideo(output_path,30,1920,1080)
                    scale_x=640/1920
                    scale_y=480/1080                    
                initial_bbox_scaled=[int(initial_bbox[0]*scale_x),int(initial_bbox[1]*scale_y),int(initial_bbox[2]*scale_x),int(initial_bbox[3]*scale_y)]
                tracker.initialize(frame, initial_bbox)
                crop_init=frame[initial_bbox[1]:initial_bbox[1]+initial_bbox[3],initial_bbox[0]:initial_bbox[0]+initial_bbox[2],:]

                feature_init = extract_features(crop_init)                
                frame_init=frame.copy()                
                start=time.time()
                
            else:
                #print(time.time()-start_skipped)
                # copy state
                state,template_features=tracker.get_state()
                state_old=copy.deepcopy(state)
                template_features_old=copy.deepcopy(template_features.detach())                

                tracked_bbox, score = tracker_utils.track_frame(tracker, frame)
                
                x, y, w, h = tracked_bbox
                
                crop=frame[y:y+h,x:x+w]        
                # blur estimation
                blur=np.round(myUtils.variance_of_laplacian(frame),2)
                if (time.time()-start)>=3:
                    if blur>20:
                        noReID=False
                        #print("time passed: ", (time.time()-start))
                
                if score>0.999 and noReID==True :
                    skipped=0
                    num_blurred=0
                    x, y, w, h = initial_bbox
                    crop_init=frame_init[y:y+h,x:x+w]               
                        
                    frame_c=tracker_utils.draw_bbox(frame_c, tracked_bbox)
                    # cv2.putText(frame_c,  "BLUR: "+str(blur)+" score "+str(np.round(score,4)), (150, 150),
                    # cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)                    
                    video.write(frame_c)
                    
                elif noReID==False:                    
                    frame,initial_bbox_t=re_id.runReID(tracker, count, frame_init, frame, initial_bbox_scaled, feature_init,scale_x, scale_y, device, extractor, matcher, predictor, vertical=vertical)
                    if initial_bbox_t is not None:
                        initial_bbox=initial_bbox_t
                    video.write(frame)
                    start=time.time()
                    noReID=True
                                    
                elif score<=0.999 and noReID==True : 
                    num_blurred=num_blurred+1                    
                    
                    x, y, w, h = tracked_bbox                    
                    crop=frame[y:y+h,x:x+w]

                    sim=np.round(similarity(feature_init,crop),2)
                        
                    if sim<0.65:
                        skipped=skipped+1
                        
                        if (time.time()-start_skipped) > 10:
                            
                            initial_bbox, labeled_frame_num=gui_video.run_app(video_path=video_path, count_frame=count)

                            if labeled_frame_num>=count:
                                for ff in range(0,((labeled_frame_num)-count)):
                                    ret, frame = cap.read()
                                    video.write(frame)                                
                            count=labeled_frame_num                            
                            ret, frame = cap.read() 
                            # cv2.putText(frame, "Frame "+str(count)+" Manual Re-ID", (150, 150),
                            # cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 3)
                            frame=tracker_utils.draw_bbox(frame, initial_bbox, color=(255,0,255))
                            video.write(frame)   
                            
                            tracker.reset()        
                            tracker.initialize(frame, initial_bbox)
                            skipped=0
                            start_skipped=time.time()

                        elif skipped>=3 and blur>20: 
                            response, initial_bbox_t=re_id.features_match(frame_init, frame,initial_bbox_scaled, feature_init, scale_x, scale_y, device, extractor, matcher, predictor, vertical=vertical)                            
                            if response==False:
                                # cv2.putText(frame, "Re-ID Failed", (150, 150),
                                # cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 3)
                                video.write(frame)
                            else:
                                if myUtils.checkBbox(initial_bbox_t, frame.shape[1], frame.shape[0]):
                                    initial_bbox=initial_bbox_t
                                    tracker.reset()
                                    skipped=0
                                    start_skipped=time.time()

                                    tracker.initialize(frame, initial_bbox)

                                    frame=tracker_utils.draw_bbox(frame, initial_bbox_t)
                                    # cv2.putText(frame, "Re-ID", (150, 150),
                                    #     cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 3)
                                    
                                    video.write(frame)
                                else:
                                    # cv2.putText(frame, "Nothing to Re-ID", (150, 150),
                                    # cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 3)
                                    video.write(frame)
                        else:
                            tracker.set_state(state_old,template_features_old)                            
                            video.write(frame)

                    else:
                        skipped=0
                        start_skipped=time.time()
                        # cv2.putText(frame_c, "BLUR: "+str(blur)+" SIM: "+str(sim)+" score "+str(np.round(score,4)), (150, 150),
                        #         cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                        frame_c=tracker_utils.draw_bbox(frame_c, tracked_bbox)
                        video.write(frame_c)
    
    # When everything done, release the video capture object
    cap.release()
    video.release()
    
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='tracking routine')        
    #parser.add_argument('--video_path',type=str, help='video_path', required=True)
    #parser.add_argument('--video_path',type=str, help='video_path', default="/home/vc/Documents/EDGE/out_PXL_20240614_054927321.TS.mp4", required=False)
    #parser.add_argument('--video_path',type=str, help='video_path', default="/home/vc/Documents/EDGE/PXL_20240614_054110286.TS.mp4", required=False)
    parser.add_argument('--video_path',type=str, help='video_path', default="/home/vc/Documents/EDGE/PXL_20240614_055315498.TS.mp4", required=False)
    parser.add_argument('--config_path',type=str, help='video_path', default="FEARTracker/model_training/config", required=False)
    parser.add_argument('--config_name',type=str, help='video_path', default="fear_tracker", required=False)    
    parser.add_argument('--weights_path',type=str, help='video_path', default="FEARTracker/evaluate/checkpoints/FEAR-XS-NoEmbs.ckpt", required=False)    

    args = parser.parse_args()

    bbox=gui.run_app(video=args.video_path)     
    
    bbox=[bbox[0]-int(bbox[2]*0.1),bbox[1]-int(bbox[3]*0.1),bbox[2]+int(bbox[2]*0.1),bbox[3]+int(bbox[3]*0.1)]    
    main(initial_bbox=bbox,video_path=args.video_path,config_path=args.config_path,config_name=args.config_name,weights_path=args.weights_path)
