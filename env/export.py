import ultralytics
import tqdm

models = ["yolo26n.pt", "yolo26s.pt"]

for name in tqdm.tqdm(models):
    model = ultralytics.YOLO(name)
    model.export(format="onnx", simplify=True)
