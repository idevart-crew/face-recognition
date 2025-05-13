from scipy.spatial.distance import cosine
import os

class FaceCompareService:
    @staticmethod
    def process(emb1, emb2):
        similarity = round((1 - cosine(emb1, emb2)) * 100, 2)
        result = True
        if (similarity < float(os.getenv("FACE_SIMILARITY_THRESHOLD"))):
            result = False

        return {"status": True, "message": "ok", "data": {"result": result, "similarity": f"{similarity:.2f}%"}}
