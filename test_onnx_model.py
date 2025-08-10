import onnxruntime as ort
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load ONNX model
session = ort.InferenceSession("yolov8n.onnx")

# Load and preprocess the image
img = cv2.imread("test_lawn.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img_rgb, (640, 640))
img_input = img_resized.transpose(2, 0, 1).astype(np.float32) / 255.0
img_input = np.expand_dims(img_input, axis=0)

# Run inference
inputs = {session.get_inputs()[0].name: img_input}
outputs = session.run(None, inputs)[0][0]

# Parse predictions
boxes = outputs[:, :4]
scores = outputs[:, 4]
class_ids = outputs[:, 5].astype(int)

# Filter by confidence threshold
CONFIDENCE_THRESHOLD = 0.5
keep = scores > CONFIDENCE_THRESHOLD
boxes = boxes[keep]
scores = scores[keep]
class_ids = class_ids[keep]

# Rescale boxes to original image size
h_orig, w_orig = img.shape[:2]
scale_x = w_orig / 640
scale_y = h_orig / 640

overlay = img.copy()

for box, score, class_id in zip(boxes, scores, class_ids):
    x1, y1, x2, y2 = box
    x1 = int(x1 * scale_x)
    y1 = int(y1 * scale_y)
    x2 = int(x2 * scale_x)
    y2 = int(y2 * scale_y)

    # Draw box
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Prepare text
    label = f"{class_id} {score:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    thickness = 1
    color = (0, 0, 0)  # black

    # Get size and place text
    (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
    text_x, text_y = x1, y1 - 5 if y1 - 5 > 10 else y1 + 15

    cv2.putText(overlay, label, (text_x, text_y), font, font_scale, color, thickness)

# Apply transparency
alpha = 0.5  # 0 = invisible, 1 = solid
output_img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

# Display result
plt.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
plt.title("Detection Results")
plt.axis("off")
plt.show()
