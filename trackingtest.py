from ultralytics import YOLO
model = YOLO("yolo11m-pose.pt") #m can handle people who are on the floor and in weird positions
results = model.track(source="c:/Users/ethomas308/Documents/CompVision/trip12fps.mp4", show=False, save=True)
print(results)

