import torch
import cv2
import os
from pathlib import Path


def load_model(model_path):
    print("Loading YOLOv5 model from local file...")
    if not os.path.exists(model_path):
        available_files = os.listdir(os.path.dirname(model_path))
        raise FileNotFoundError(f"Model file not found at {model_path}. Available files: {available_files}")
    model = torch.load(model_path)  # Load the model directly
    model.eval()  # Set the model to evaluation mode
    print("Model loaded successfully.")
    return model


def process_video(video_path, model, output_dir, results_file):
    print(f"Processing video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out_path = os.path.join(output_dir, Path(video_path).stem + '_output.mp4')
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to a tensor and add a batch dimension
        img_tensor = torch.from_numpy(frame).float().permute(2, 0, 1).unsqueeze(0)
        results = model(img_tensor)
        results = results.xyxy[0]  # Extract the results for the current frame

        # Draw bounding boxes on the frame
        for *box, conf, cls in results:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()

    with open(results_file, 'a') as f:
        f.write(f"Video: {video_path}\n")
        f.write(f"Total frames: {frame_count}\n")
        f.write(f"Output saved to: {out_path}\n")
        f.write("\n")

    print(f"Finished processing {video_path}. Total frames: {frame_count}. Output saved to {out_path}")


def process_directory(input_dir, output_dir, model, results_file):
    for subdir, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(('.mp4', '.avi', '.mov')):
                video_path = os.path.join(subdir, file)
                try:
                    process_video(video_path, model, output_dir, results_file)
                except Exception as e:
                    with open(results_file, 'a') as f:
                        f.write(f"Error processing {video_path}: {e}\n")
                    print(f"Error processing {video_path}: {e}")


if __name__ == "__main__":
    model_path = '/Users/pedro/Desktop/crash_detection_cnn/scripts/yolov5s.pt'
    model = load_model(model_path)

    # Define paths
    training_videos_dir = '/Users/pedro/Desktop/crash_detection_cnn/data/videos/training'
    testing_videos_dir = '/Users/pedro/Desktop/crash_detection_cnn/data/videos/testing'
    output_dir = '/Users/pedro/Desktop/crash_detection_cnn/scripts/runs/detect'
    training_results_file = '/Users/pedro/Desktop/crash_detection_cnn/scripts/runs/training_results.txt'
    testing_results_file = '/Users/pedro/Desktop/crash_detection_cnn/scripts/runs/testing_results.txt'

    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)

    # Clear previous results files
    open(training_results_file, 'w').close()
    open(testing_results_file, 'w').close()

    # Process training videos
    print("Processing training videos...")
    process_directory(training_videos_dir, output_dir, model, training_results_file)
    print("Finished processing training videos.")

    # Process testing videos
    print("Processing testing videos...")
    process_directory(testing_videos_dir, output_dir, model, testing_results_file)
    print("Finished processing testing videos.")

    print("Processing completed.")
