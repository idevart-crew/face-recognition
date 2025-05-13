import base64
import io
import os
from dotenv import load_dotenv
from PIL import Image
from fastapi.responses import JSONResponse

load_dotenv()
MAX_WIDTH = float(os.getenv("MAX_IMAGE_WIDTH"))
MAX_HEIGHT = float(os.getenv("MAX_IMAGE_HEIGHT"))

def validate_image_ratio(base64_string: str):
    try:
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        width, height = image.size

        if round(width / height, 2) != round(3 / 4, 2):
            return {"status": False, "message": f"Invalid image ratio. Expected 3:4."}

        if width > MAX_WIDTH or height > MAX_HEIGHT:
            return {"status": False, "message": f"Image too large ({width}x{height}). Max allowed: {MAX_WIDTH}x{MAX_HEIGHT}."}

        return {"status": True, "message": "Valid image"}

    except Exception as e:
        return {"status": False, "message": f"Invalid image file: {str(e)}"}
