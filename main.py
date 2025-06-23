import cv2
from ultralytics import YOLO
import numpy as np
import csv

# Simple centroid-based tracker
class CentroidTracker:
    def __init__(self, max_disappeared=30):
        self.next_object_id = 0
        self.objects = dict()
        self.disappeared = dict()
        self.max_disappeared = max_disappeared

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, rects):
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            input_centroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(0, len(input_centroids)):
                self.register(input_centroids[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())
            D = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - input_centroids, axis=2)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            used_rows = set()
            used_cols = set()
            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0
                used_rows.add(row)
                used_cols.add(col)
            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)
            if D.shape[0] >= D.shape[1]:
                for row in unused_rows:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            else:
                for col in unused_cols:
                    self.register(input_centroids[col])
        return self.objects

# Load YOLOv11 model
model = YOLO('best.pt')

# Open the video file
cap = cv2.VideoCapture('15sec_input_720p.mp4')

if not cap.isOpened():
    print('Error: Could not open video file.')
    exit()

# Get video properties for saving output
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_tracked.mp4', fourcc, fps, (width, height))

tracker = CentroidTracker(max_disappeared=30)

# Prepare CSV for saving tracking results
csv_file = open('tracking_results.csv', mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['frame', 'object_id', 'x1', 'y1', 'x2', 'y2', 'centroid_x', 'centroid_y'])

frame_idx = 0

# Get class names from the model (assume 'player' is class 0 or find correct index)
class_names = model.names if hasattr(model, 'names') else ['player', 'referee', 'ball']
# Ensure class_names is a list of strings
class_names = [str(name) for name in class_names]
player_class_indices = [i for i, name in enumerate(class_names) if 'player' in name.lower()]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model(frame)
    boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes is not None else []
    classes = results[0].boxes.cls.cpu().numpy() if results[0].boxes is not None else []
    rects = []
    filtered_boxes = []
    for box, cls in zip(boxes, classes):
        if int(cls) in player_class_indices:
            x1, y1, x2, y2 = box.astype(int)
            rects.append((x1, y1, x2, y2))
            filtered_boxes.append((x1, y1, x2, y2))

    # Update tracker only with player boxes
    objects = tracker.update(rects)

    # Visualize results and save to CSV
    for ((x1, y1, x2, y2), centroid) in zip(filtered_boxes, objects.values()):
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        object_id = None
        for oid, c in objects.items():
            if np.array_equal(c, centroid):
                object_id = oid
                break
        if object_id is not None:
            cv2.putText(frame, f'ID {object_id}', (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.circle(frame, tuple(centroid), 4, (0, 255, 0), -1)
            # Save to CSV
            csv_writer.writerow([frame_idx, object_id, x1, y1, x2, y2, centroid[0], centroid[1]])

    out.write(frame)
    cv2.imshow('YOLOv11 Player Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    frame_idx += 1

cap.release()
out.release()
csv_file.close()
cv2.destroyAllWindows()

print('Tracking complete. Output video: output_tracked.mp4, CSV: tracking_results.csv') 