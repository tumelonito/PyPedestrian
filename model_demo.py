import cv2
import torch
import torchvision

# --- Config ---
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
NUM_WORKERS = 2 if torch.cuda.is_available() else 0
MODEL_FILENAME = 'quant_model.pt'
CONFIDENCE_THRESHOLD = 0.5
PROCESS_EVERY_N_FRAMES = 2
MODEL_INPUT_SIZE = (480, 360)

print(f"[INFO] Current device: {DEVICE}")
if DEVICE.type == 'cpu':
    print("[WARNING] GPU not detected.")


# --- Loading the model ---
model = torch.jit.load(MODEL_FILENAME, map_location='cpu')
model.to(DEVICE)
model.eval()


# --- Camera initializing ---
frame_counter = 0
last_boxes = []
last_scores = []
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Couldn't open camera.")
    exit()

print("Camera started successfully. Press 'Q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display_frame = frame.copy()

    # Every n frame do:
    if frame_counter % PROCESS_EVERY_N_FRAMES == 0:

        # Frame preparation
        resized_frame = cv2.resize(frame, MODEL_INPUT_SIZE)
        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        img_tensor = torchvision.transforms.functional.to_tensor(rgb_frame).to(DEVICE)

        # --- Prediction ---
        with torch.no_grad():
            # For JIT models, a list of tensors must be passed
            output = model([img_tensor])

            # Note: TorchScript sometimes returns a tuple (loss, predictions)
            # extracts the dictionary containing the predictions
            predictions = output[1] if isinstance(output, tuple) else output

            if isinstance(predictions, list):
                prediction = predictions[0]
            else:
                prediction = predictions

        # --- Prediction processing ---
        orig_h, orig_w = frame.shape[:2]
        scale_x = orig_w / MODEL_INPUT_SIZE[0]
        scale_y = orig_h / MODEL_INPUT_SIZE[1]

        last_boxes = []
        last_scores = []

        boxes = prediction['boxes'].cpu().numpy()
        scores = prediction['scores'].cpu().numpy()

        for box, score in zip(boxes, scores):
            if score >= CONFIDENCE_THRESHOLD:
                x1, y1, x2, y2 = box
                last_boxes.append([int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)])
                last_scores.append(score)

    frame_counter += 1

    # --- Visualizing ---
    for box, score in zip(last_boxes, last_scores):
        x1, y1, x2, y2 = box
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"Pedestrian: {score * 100:.1f}%"
        cv2.putText(display_frame, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Pedestrian Detection (Press 'Q' to exit)", display_frame)

    # Exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()