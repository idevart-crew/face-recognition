from insightface.app import FaceAnalysis
import numpy as np
import cv2
import base64
import os

class FaceRecognitionService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FaceRecognitionService, cls).__new__(cls)
            # cls._instance.face_app = FaceAnalysis(name=os.getenv("FACE_RECOGNITION_MODEL"), providers=["CPUExecutionProvider"])
            cls._instance.face_app = FaceAnalysis(name=os.getenv("FACE_RECOGNITION_MODEL"), providers=["CUDAExecutionProvider"])
            cls._instance.face_app.prepare(ctx_id=0)
        return cls._instance

    async def get_embedding(self, base64_string: str):
        image_data = base64.b64decode(base64_string)
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        faces = self.face_app.get(img)

        if len(faces) == 0:
            return {"status": False, "message": "No face detected", "data": None}

        largest_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        
        # Tilt face detection
        # landmarks = largest_face.landmark_3d_68
        # left_eye = landmarks[36] 
        # right_eye = landmarks[45]
        # eye_diff = float(f"{abs(left_eye[1] - right_eye[1]):.2f}")
        # if eye_diff > float(os.getenv("FACE_TILT_TOLERANCE")):
        #     return {
        #         "status": False, 
        #         "message": "Face is asymmetrical/tilted.", 
        #         "data": {
        #             "tilt": f"{eye_diff} degree",
        #             "conf": float(f"{largest_face.det_score:.2f}")
        #         }
        #     }
            
        return {
            "status": True, 
            "message": "ok", 
            "data": {
                "img" : img,
                "embd": largest_face.normed_embedding, 
                "bbox": self.make_bbox_square(largest_face.bbox, img.shape, 0), 
                "conf": float(f"{largest_face.det_score:.2f}"),
                # "tilt": f"{eye_diff} degree"
            }
        }

    def make_bbox_square(self, bbox, img_shape, margin_ratio=0.1):
        x1, y1, x2, y2 = bbox.astype(int)
        w = x2 - x1
        h = y2 - y1

        side = h
        center_x = x1 + w // 2
        offset_y = int(0.1 * h)
        new_x1 = center_x - side // 2
        new_y1 = y1 - offset_y

        margin = int(margin_ratio * side)
        new_x1 -= margin
        new_y1 -= margin
        side += 2 * margin

        new_x1 = max(new_x1, 0)
        new_y1 = max(new_y1, 0)
        new_x1 = min(new_x1, img_shape[1] - side)
        new_y1 = min(new_y1, img_shape[0] - side)

        new_x2 = new_x1 + side
        new_y2 = new_y1 + side

        return np.array([new_x1, new_y1, new_x2, new_y2])


face_service_instance = FaceRecognitionService()
def get_face_service():
    return face_service_instance
