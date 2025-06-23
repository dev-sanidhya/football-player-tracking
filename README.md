# Football Player Detection and Re-Identification

This project uses a fine-tuned YOLOv11 model to detect and re-identify football players in a video. The solution tracks players with consistent IDs, saves an annotated output video, and exports tracking results to a CSV file.

## Setup

1. **Clone this repository or download the files.**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Download the YOLOv11 model weights:**
   - Download `best.pt` from the original assignment link: [Google Drive - best.pt](https://drive.google.com/file/d/1-5f0SHOSB9UXvP_enOoZNAMScrePVcMD/view)
   - Place `best.pt` in the project directory.
4. **Ensure the following file is in the project directory:**
   - `15sec_input_720p.mp4` (input video)

## Usage

Run the main script to perform detection, tracking, and save results:
```bash
python main.py
```
- Press `q` to quit the video window.
- Output video: `output_tracked.mp4` (generated after running, not included in repo)
- Tracking CSV: `tracking_results.csv` (generated after running, not included in repo)
- Brief report: `REPORT.md`

## Features & Improvements
- **Player-only tracking:** Only players are tracked and assigned IDs (referees and balls are ignored).
- **Consistent re-identification:** Players keep the same ID even if they leave and re-enter the frame.
- **Output video and CSV:** Results are saved for reproducibility and further analysis.
- **Self-contained and documented:** All code and instructions are included.

## Notes on Repository Contents
- The model weights file (`best.pt`) is **not included** in this repository due to file size limits. Please download it from the provided Google Drive link.
- The output video (`output_tracked.mp4`) and tracking CSV (`tracking_results.csv`) are **not included** in the repository. You can generate them by running the script as described above.

## Troubleshooting
- If you see import errors for `cv2` or `ultralytics`, ensure you have installed all dependencies with the correct Python environment.
- If the video does not open, check the filename and ensure the video file is in the same directory as the script.

## Next Steps
- For more robust tracking, consider integrating a tracker like SORT or DeepSORT.
- See `REPORT.md` for a summary of the approach, techniques, and challenges. 