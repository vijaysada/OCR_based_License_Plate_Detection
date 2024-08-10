from ultralytics import YOLO
import argparse
import os
import cv2
import numpy as np
import torch
from torchvision import transforms
import shutil
import asyncio 
import time
import sys
import multiprocessing
from vidgear.gears import CamGear
import pandas as pd
from paddleocr import PaddleOCR

# Initialize the PaddleOCR object
ocr = PaddleOCR()

# # Path to the image file
# image_path = r"C:\Users\DELL E5490\Downloads\01.jpg"




def model_defn(act_model_path):
    act_model = YOLO(act_model_path)
    print("model loaded successfully")
    return act_model

# def detect_video_parallel(act_model,video_sources,dest_path):
#     # Set the start method to 'spawn' to avoid CUDA conflicts

#     multiprocessing.set_start_method('spawn')
#     processes = [] #0,1

#     for i, video_src in enumerate(video_sources):
#         process = multiprocessing.Process(target=detect_video, args=(act_model, video_src,i,dest_path))
#         processes.append(process)
#         process.start()

#     for process in processes:
#         process.join()

def detect_video(act_model, video_src, dest_path):
    # Open the video file

    cap = cv2.VideoCapture(video_src)
    if not cap.isOpened():
        raise ValueError(f"Error opening video file: {video_src}")

    process_output_dir = os.path.join(dest_path, f"process")
    os.makedirs(process_output_dir, exist_ok=True)

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the codec and create VideoWriter object
    output_video_path = os.path.join(dest_path, 'output_video.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    fr_cnt = 0
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break


        results = act_model.predict(frame, conf=0.30, save_crop=False)
        save_crop(results, frame, count, process_output_dir,out)
        count += 1
    
        
        fr_cnt += 1
        out.write(frame)

    cap.release()

def save_crop(results, frame, frame_number, process_output_dir,out):
    coordinates = []
    texts = []
    cnt = 0
    if len(results[0]) >= 1:
        if (results[0].boxes.cls.cpu().numpy()[0] == 0) or (results[0].boxes.cls.cpu().numpy()[0] == 1):
            for c_xywh in results[0].boxes.xyxy:
                c_xywh = c_xywh.cpu()
                x = int(c_xywh.numpy()[0])
                y = int(c_xywh.numpy()[1])
                w = int(c_xywh.numpy()[2] - x)
                h = int(c_xywh.numpy()[3] - y)
                coordinates.append([x,y,w,h])

                # Draw bounding box
                color = (0, 0, 255)  
                thickness = 2
                frame = cv2.rectangle(frame, (x, y), (w+x, h+y), color, thickness)

                image_res = frame[y:y+h, x:x+w]
                cv2.imwrite(f"./ffff/{frame_number}.jpg",frame)
                # new_width = int(image_res.shape[1] * 3)
                # new_height = int(image_res.shape[0] * 3)
                # resized_image = cv2.resize(image_res, (new_width, new_height))
                result = ocr.ocr(image_res, cls=True)

                for i in range(0,len(result)):
                    text = result[0][0][1][0]
                    text_position = (x, y - 10)  # Position the text above the box
                    frame = cv2.putText(frame, f"{text}", text_position, cv2.FONT_HERSHEY_SIMPLEX,2, color, 3, cv2.LINE_AA)
                    cv2.imwrite(f"./ffff/{frame_number}.jpg",frame)
                    out.write(frame)
    
            
            
if __name__ == "__main__":
    #argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_paths',nargs='+', type = str, required = False) 
    parser.add_argument('--dest_path', type = str, required = True)
    parser.add_argument('--act_model_path', type = str, default = "license_plate_detector.pt", required = False)

    args = parser.parse_args()
    isExist = os.path.exists(args.dest_path)
    
    if not isExist:
        os.makedirs(args.dest_path)
    
    act_model = model_defn(args.act_model_path)
    video_src = args.source_paths
    
    detect_video(act_model, r"cars.mp4", args.dest_path)
