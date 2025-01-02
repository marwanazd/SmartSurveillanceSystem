from ultralytics import YOLO



model = YOLO('models/knive.pt')
model.export(format='onnx')
