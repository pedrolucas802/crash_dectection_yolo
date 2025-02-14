import torch

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

imgs = ['/Users/pedro/Desktop/crash_dectection_cnn/data/test/Accident/test1_24.jpg']

results = model(imgs)

results.print()

# results.save()

results.show()

print(results.xyxy[0])

print(results.pandas().xyxy[0])
