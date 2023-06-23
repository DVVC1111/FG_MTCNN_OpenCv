import cv2
from PIL import Image, ImageOps
from facenet_pytorch import MTCNN
import numpy as np
import dlib

def fix_image_orientation(image):
    return ImageOps.exif_transpose(image)

def equalize_histogram(image):
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    channels = cv2.split(ycrcb)
    cv2.equalizeHist(channels[0], channels[0])
    cv2.merge(channels, ycrcb)
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_Lab2BGR)

def check_landmarks(face_rect, predictor, image_gray):
    landmarks = predictor(image_gray, face_rect)
    if len(landmarks.parts()) == 68:
        return True
    else:
        return False

def detect_faces(detector, landmark_predictor):
    cap = cv2.VideoCapture(0) # Use 0 for default webcam, or replace with camera index

    while True:
        ret, frame = cap.read() 
        

        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image)
        image_pil = fix_image_orientation(image_pil)

        image_cv2 = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        image_preprocessed = image_cv2.copy()
        image_preprocessed = equalize_histogram(image_preprocessed)
        image_preprocessed = apply_clahe(image_preprocessed)
        image_gray = cv2.cvtColor(image_preprocessed, cv2.COLOR_BGR2GRAY)

        bboxes, _ = detector.detect(cv2.cvtColor(image_preprocessed, cv2.COLOR_BGR2RGB))

        if bboxes is not None:
            for bbox in bboxes:
                x_min, y_min, x_max, y_max = map(int, bbox[:4])
                face_rect = dlib.rectangle(x_min, y_min, x_max, y_max)

                if check_landmarks(face_rect, landmark_predictor, image_gray):
                    cv2.rectangle(image_cv2, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        cv2.imshow("Detected Faces", image_cv2)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    detector = MTCNN(thresholds=[0.8, 0.9, 0.9]) 

    landmark_predictor_path = "shape_predictor_68_face_landmarks.dat"
    landmark_predictor = dlib.shape_predictor(landmark_predictor_path)

    detect_faces(detector, landmark_predictor)

if __name__ == "__main__":
    main()