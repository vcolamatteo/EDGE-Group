import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sam import loadPredictor, segment


class VideoClickApp:
    def __init__(self, root, video_path, count_frame):
        self.root = root
        self.root.title("Object Selector GUI")

        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.current_frame_index = count_frame

        self.predictor=loadPredictor()

        self.bbox = []
        self.points = []
        self.point_ids = []

        self.load_frame(self.current_frame_index)

        self.canvas = tk.Canvas(root, width=self.img.width(), height=self.img.height())
        self.canvas.pack()
        self.image_on_canvas = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img)

        self.canvas.bind("<Button-1>", self.get_coordinates)
        self.canvas.bind("<Motion>", self.show_cursor_position)
        #self.canvas.bind("<Button-1>", self.show_frame_index)

        self.coord_label = tk.Label(root, text="Clicked at: (None, None)")
        self.coord_label.pack()
        self.cursor_label = tk.Label(root, text="Cursor at: (None, None)")
        self.cursor_label.pack()
        self.frame_label = tk.Label(root, text=f"Current Frame: {self.current_frame_index}")
        self.frame_label.pack()

        self.back_button = tk.Button(root, text="Back", command=self.remove_last_point)
        self.back_button.pack(side=tk.LEFT)
        self.empty_button = tk.Button(root, text="Empty", command=self.remove_all_points)
        self.empty_button.pack(side=tk.LEFT)
        self.done_button = tk.Button(root, text="Done", command=self.done)
        self.done_button.pack(side=tk.LEFT)

        self.next_frame_button = tk.Button(root, text="Next Frame", command=self.next_frame)
        self.next_frame_button.pack(side=tk.LEFT)
        self.prev_frame_button = tk.Button(root, text="Prev Frame", command=self.prev_frame)
        self.prev_frame_button.pack(side=tk.LEFT)

        self.exit_button = tk.Button(root, text="Exit", command=self.exit)
        self.exit_button.pack(side=tk.LEFT)

        # Hidden "Ok" and "Retry" buttons
        self.ok_button = tk.Button(root, text="Ok", command=self.ok)
        self.retry_button = tk.Button(root, text="Retry", command=self.retry)
        self.exit_button = tk.Button(root, text="Exit", command=self.ok)

    def load_frame(self, count_frame, prev_frame=0):
        if prev_frame==0:
            self.cap = cv2.VideoCapture(self.video_path)
        # Move to the desired frame
        current_frame = prev_frame        
        #print(count_frame,current_frame)
        while current_frame < count_frame:
            ret, frame = self.cap.read()            
            if not ret:
                print("Reached end of video or failed to read the frame.")
                return
            current_frame += 1            

        ret, frame = self.cap.read()
        if ret:
            if frame.shape[0]>frame.shape[1]:
                frame=cv2.resize(frame,(480,640))                
            else:
                frame=cv2.resize(frame,(1280,720))
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            self.original_image = Image.fromarray(frame)            
            self.image_copy = self.original_image.copy()
            self.img = ImageTk.PhotoImage(self.original_image)
            if hasattr(self, 'frame_label'):
                self.frame_label.config(text=f"Current Frame: {self.current_frame_index}")

    def get_coordinates(self, event):
        x, y = event.x, event.y
        self.coord_label.config(text=f"Clicked at: ({x}, {y})")
        #print(f"Clicked at: ({x},F {y})")
        
        radius = 5
        point_id = self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, outline="green", width=2, fill="green")
        self.points.append((x, y))
        self.point_ids.append(point_id)

    def show_cursor_position(self, event):
        x, y = event.x, event.y
        self.cursor_label.config(text=f"Cursor at: ({x}, {y})")
    
    def show_frame_index(self, event):
        self.frame_label.config(text=f"Frame number: "+str(self.current_frame_index))

    def remove_last_point(self):
        if self.point_ids:
            last_id = self.point_ids.pop()
            self.canvas.delete(last_id)
            self.points.pop()
            #print("Last point removed")

    def remove_all_points(self):
        while self.point_ids:
            self.canvas.delete(self.point_ids.pop())
        self.points.clear()
        #print("All points removed")

    def done(self):
        #print("Selected points:", self.points)
        self.segment(self.image_copy, self.points, self.predictor)
        
    def segment(self, image, points, predictor):
        image, bbox = segment(np.array(image), np.array(points), predictor)
        self.bbox = bbox
        self.edges_img = ImageTk.PhotoImage(image)
        self.canvas.itemconfig(self.image_on_canvas, image=self.edges_img)

        # Hide previous buttons and labels
        self.coord_label.pack_forget()
        self.cursor_label.pack_forget()
        self.back_button.pack_forget()
        self.empty_button.pack_forget()
        self.done_button.pack_forget()
        self.prev_frame_button.pack_forget()
        self.next_frame_button.pack_forget()


        # Add "Ok" and "Retry" buttons
        self.ok_button.pack(side=tk.LEFT)
        self.retry_button.pack(side=tk.LEFT)

    def ok(self):
        #print("Selected points:", self.points)
        plt.close()
        self.root.destroy()
        self.root.quit()

    def exit(self):
        plt.close()
        self.cap.release()
        self.root.destroy()
        self.root.quit()

    def retry(self):
        # Reset the canvas with the original image
        self.canvas.itemconfig(self.image_on_canvas, image=self.img)
        self.points.clear()
        self.point_ids.clear()

        self.image_on_canvas = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img)
        
        # Restore "Back", "Empty", and "Done" buttons
        self.coord_label.config(text="Clicked at: (None, None)")
        self.coord_label.pack()
        self.cursor_label.config(text="Cursor at: (None, None)")
        self.cursor_label.pack()
        self.back_button.pack(side=tk.LEFT)
        self.empty_button.pack(side=tk.LEFT)
        self.done_button.pack(side=tk.LEFT)
        self.prev_frame_button.pack(side=tk.LEFT)
        self.next_frame_button.pack(side=tk.LEFT)

        # Remove "Ok" and "Retry" buttons
        self.ok_button.pack_forget()
        self.retry_button.pack_forget()
        self.exit_button.pack_forget()
        
    def next_frame(self):
        self.prev_frame_index = self.current_frame_index        
        self.load_frame(self.current_frame_index, self.prev_frame_index)
        self.current_frame_index += 1
        self.canvas.itemconfig(self.image_on_canvas, image=self.img)
        self.frame_label.config(text=f"Current Frame: {self.current_frame_index}")

    def prev_frame(self):
        
        #self.prev_frame_index = self.current_frame_index                    
        if self.current_frame_index > 0:
            self.current_frame_index -= 1
        self.load_frame(self.current_frame_index)
        self.canvas.itemconfig(self.image_on_canvas, image=self.img)
        self.frame_label.config(text=f"Current Frame: {self.current_frame_index}")


def run_app(video_path, count_frame):
    root = tk.Tk()
    app = VideoClickApp(root, video_path, count_frame)
    root.mainloop()
    

    if np.array(app.original_image).shape[0] > np.array(app.original_image).shape[1]:
        scale_y=1920/640
        scale_x=1080/480                
    else:        
        scale_x=1920/1270
        scale_y=1080/720
    
    return [int(app.bbox[0]*scale_x),int(app.bbox[1]*scale_y),int(app.bbox[2]*scale_x),int(app.bbox[3]*scale_y)], app.current_frame_index
    


# if __name__ == "__main__":
#     root = tk.Tk()
#     video_path = "/home/vc/Documents/EDGE/PXL_20240614_055315498.TS.mp4"  # Replace with your video file path
#     player = VideoClickApp(root, video_path,count_frame=180)
#     root.mainloop()

#     print(player.bbox, player.current_frame_index)
