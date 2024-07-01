import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sam import loadPredictor, segment

class ImageClickApp:
    def __init__(self, root, file_path):
        self.root = root
        self.root.title("Object Selector GUI")

        # Initialize points list
        self.bbox=[]
        self.points = []
        self.point_ids = []

        self.predictor=loadPredictor()

        # Load image
        self.load_image(file_path)

        # Display image
        self.canvas = tk.Canvas(root, width=self.img.width(), height=self.img.height())
        self.canvas.pack()
        self.image_on_canvas = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img)

        # Bind events
        self.canvas.bind("<Button-1>", self.get_coordinates)
        self.canvas.bind("<Motion>", self.show_cursor_position)

        # Labels and buttons
        self.coord_label = tk.Label(root, text="Clicked at: (None, None)")
        self.coord_label.pack()
        self.cursor_label = tk.Label(root, text="Cursor at: (None, None)")
        self.cursor_label.pack()

        self.back_button = tk.Button(root, text="Back", command=self.remove_last_point)
        self.back_button.pack(side=tk.LEFT)
        self.empty_button = tk.Button(root, text="Empty", command=self.remove_all_points)
        self.empty_button.pack(side=tk.LEFT)
        self.done_button = tk.Button(root, text="Done", command=self.done)
        self.done_button.pack(side=tk.LEFT)
        self.exit_button = tk.Button(root, text="Exit", command=self.exit)
        self.exit_button.pack(side=tk.LEFT)

        # Hidden "Ok" and "Retry" buttons
        self.ok_button = tk.Button(root, text="Ok", command=self.ok)
        self.retry_button = tk.Button(root, text="Retry", command=self.retry)
        self.exit_button = tk.Button(root, text="Exit", command=self.ok)

    def load_image(self, file):

        self.original_image = file
        #self.original_image = Image.open(file_path)
        self.image_copy = self.original_image.copy()
        self.img = ImageTk.PhotoImage(self.original_image)

    def get_coordinates(self, event):
        x, y = event.x, event.y
        self.coord_label.config(text=f"Clicked at: ({x}, {y})")
        #print(f"Clicked at: ({x}, {y})")
        
        # Draw a green circle at the clicked point and save the point and canvas ID
        radius = 5
        point_id = self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, outline="green", width=2, fill="green")
        self.points.append((x, y))
        self.point_ids.append(point_id)

    def show_cursor_position(self, event):
        x, y = event.x, event.y
        self.cursor_label.config(text=f"Cursor at: ({x}, {y})")

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

    def cancel_selection(self):
        self.coord_label.config(text="Clicked at: (None, None)")
        #print("Selection canceled")

    def done(self):
        self.segment(self.image_copy, self.points, self.predictor)

    def open_new_gui(self):
        # Open a new GUI window
        self.root = tk.Tk()
        self.__init__(self.root)  # Re-initialize ImageClickApp in the new root
        self.root.mainloop()

    def segment(self, image, points, predictor):
        
        image, bbox=segment(np.array(image), np.array(points), predictor)
        self.bbox=bbox       
        self.edges_img = ImageTk.PhotoImage(image)

        # Update canvas with edges image
        self.canvas.itemconfig(self.image_on_canvas, image=self.edges_img)

        # Hide previous buttons and labels
        self.coord_label.pack_forget()
        self.cursor_label.pack_forget()
        self.back_button.pack_forget()
        self.empty_button.pack_forget()
        self.done_button.pack_forget()

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

        # Remove "Ok" and "Retry" buttons
        self.ok_button.pack_forget()
        self.retry_button.pack_forget()
        self.exit_button.pack_forget()

    def destroy_gui(self):
        # Destroy the current GUI safely from the main thread
        self.root.destroy()

def open_new_gui():
    new_root = tk.Tk()
    new_app = ImageClickApp(new_root, "result.jpg")
    new_root.mainloop()
    #print(new_app.bbox)

def run_app(video="/home/vc/Documents/EDGE/out_PXL_20240614_054927321.TS.mp4", count_frame=1):

    count=0
    #print(count_frame)
    cap = cv2.VideoCapture(video)
    # Check if camera opened successfully
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
    
    while(cap.isOpened()):
        
        # Read te first frame
        ret, frame = cap.read()        
        count=count+1
        if count==count_frame:
            if ret == True:
                if frame.shape[0]>frame.shape[1]:
                    frame=cv2.resize(frame,(480,640))
                else:
                    frame=cv2.resize(frame,(1280,720))
                
                break
        
            

    cap.release()
    
    root = tk.Tk()
    app = ImageClickApp(root, Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)))
    root.mainloop()
    print(app.bbox)
    if frame.shape[0]>frame.shape[1]:
        scale_y=1920/640
        scale_x=1080/480                
    else:        
        scale_x=1920/1270
        scale_y=1080/720
    
    return [int(app.bbox[0]*scale_x),int(app.bbox[1]*scale_y),int(app.bbox[2]*scale_x),int(app.bbox[3]*scale_y)]
        
    
    


# if __name__ == "__main__":

#     run_app()

    
    

    
