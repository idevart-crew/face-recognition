from insightface.app import FaceAnalysis
from concurrent.futures import ThreadPoolExecutor
import asyncio
import numpy as np
import cv2
import base64
import os
import time

class FaceRecognitionService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FaceRecognitionService, cls).__new__(cls)
            cls._instance.face_app = FaceAnalysis(name=os.getenv("FACE_RECOGNITION_MODEL"), root="cores", providers=["CPUExecutionProvider"])
            cls._instance.face_app.prepare(ctx_id=0)
        return cls._instance

    async def get_embedding(self, base64_string: str):
        executor = ThreadPoolExecutor(max_workers=4)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(executor, self.get_embedding_sync, base64_string)

    # async def get_embedding(self, base64_string: str):
    def get_embedding_sync(self, base64_string: str):
        image_data = base64.b64decode(base64_string)
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (256, 256))
        faces = self.face_app.get(img)

        if len(faces) == 0:
            return {"status": False, "message": "No face detected", "data": None}

        largest_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        
        # self.save_with_bbox(img, largest_face.bbox.astype(int))

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
                "embd": largest_face.normed_embedding, 
                "bbox": largest_face.bbox.astype(int), 
                "conf": float(f"{largest_face.det_score:.2f}"),
                "tilt": 0#f"{eye_diff} degree"
            }
        }

    def save_with_bbox(self, img, bbox, output_dir="output"):
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{time.time()}.jpg")

        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        cv2.imwrite(output_path, img)


face_service_instance = FaceRecognitionService()
def get_face_service():
    return face_service_instance
