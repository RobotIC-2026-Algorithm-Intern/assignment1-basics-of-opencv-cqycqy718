import cv2
import numpy as np
import json

class ColorRange:
    def __init__(self, config=None):
        self.ranges = {}
        if config is not None:
            self.load_from_config(config)

    def load_from_config(self, config):
        for color, bounds_list in config["color_ranges"].items():
            for lower, upper in bounds_list:
                self.add_hsv(color, lower, upper)

    def add_hsv(self, color_name, lower, upper):
        if color_name not in self.ranges:
            self.ranges[color_name] = []
        self.ranges[color_name].append((np.array(lower), np.array(upper)))

    def get_mask(self, hsv_img, color_name):
        mask = np.zeros(hsv_img.shape[:2], dtype=np.uint8)
        for lower, upper in self.ranges[color_name]:
            mask |= cv2.inRange(hsv_img, lower, upper)
        return mask

    def get_multi_mask(self, hsv_img, color_names):
        mask = np.zeros(hsv_img.shape[:2], dtype=np.uint8)
        for name in color_names:
            mask |= self.get_mask(hsv_img, name)
        return mask

class BallDetector:
    def __init__(self, color_range, roi=None):
        self.color_range = color_range
        self.roi = roi

    def set_roi(self, roi):
        self.roi = roi

    def detect(self, frame):
        y1, y2, x1, x2 = self.roi
        roi = frame[y1:y2, x1:x2, :]
        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV_FULL)

        red_mask = self.color_range.get_mask(roi_hsv, 'red')
        blue_mask = self.color_range.get_mask(roi_hsv, 'blue')
        purple_mask = self.color_range.get_mask(roi_hsv, 'purple')

        h, w = frame.shape[:2]
        left_third = frame[:, :w//3, :]
        left_hsv = cv2.cvtColor(left_third, cv2.COLOR_BGR2HSV_FULL)
        left_purple_mask = self.color_range.get_mask(left_hsv, 'purple')
        left_purple_ratio = np.sum(left_purple_mask > 0) / (left_third.shape[0] * left_third.shape[1])
        if left_purple_ratio > 0.1:
            return "none"

        counts = {
            'red': np.sum(red_mask > 0),
            'blue': np.sum(blue_mask > 0),
            'purple': np.sum(purple_mask > 0)
        }
        dominant = max(counts, key=counts.get)
        if counts[dominant] < 1700:
            return "none"
        return dominant

    def process_video(self, video_path, window_name="video"):
        cam = cv2.VideoCapture(video_path)
        while True:
            ret, frame = cam.read()
            if not ret:
                break
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            result = self.detect(frame)
            cv2.putText(frame, f"ball: {result}", (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
            cv2.imshow(window_name, frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        cam.release()
        cv2.destroyAllWindows()  

def main():
    with open("Homework/config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    color_range = ColorRange(config)
    roi = config["roi"]
    detector = BallDetector(color_range, roi=roi)

    detector.process_video("res/output.avi", "output.avi")
    detector.process_video("res/output1.avi", "output1.avi")

if __name__ == "__main__":
    main()