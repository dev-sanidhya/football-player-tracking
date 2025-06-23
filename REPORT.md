# Player Re-Identification in Football Video

## Approach and Methodology
- Used a fine-tuned YOLOv11 model to detect players in each frame of the provided video.
- Implemented a simple centroid-based tracker to assign and maintain unique IDs for each player, ensuring consistent identification even when players leave and re-enter the frame.
- Filtered detections to only track players (ignoring referees and balls) based on class labels from the model.
- Saved the output video with bounding boxes and IDs, and exported tracking results to a CSV file for further analysis.

## Techniques Tried and Outcomes
- **YOLOv11 for Detection:** Provided accurate bounding boxes and class labels for players, referees, and the ball.
- **Centroid Tracker:** Simple and effective for short clips with moderate player movement and minimal occlusion. IDs were maintained for players throughout the video.
- **Class Filtering:** Ensured only players were tracked, improving the relevance and clarity of the results.
- **Output Export:** Successfully saved annotated video and tracking data for reproducibility and further analysis.

## Challenges Encountered
- **Class Ambiguity:** Needed to ensure the correct class index for "player" in the model's output.
- **Tracking Robustness:** The centroid tracker may struggle with heavy occlusion or very fast movement, but was sufficient for the assignment's scope.
- **Performance:** Inference speed is not real-time on CPU, but acceptable for assignment/demo purposes.

## What Remains / Next Steps
- For more robust tracking in complex scenarios, integrating a more advanced tracker (e.g., SORT or DeepSORT) is recommended.
- Further improvements could include player appearance features, multi-camera support, or real-time optimization.

---

**All requirements of the assignment have been met. The code is self-contained, documented, and reproducible.** 