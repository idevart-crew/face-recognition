from controllers.base_controller import BaseController
from form_requests.process_request import ProcessRequest
from validators.image_validator import validate_image_ratio
from services.face_recognition_service import FaceRecognitionService
from services.face_compare_service import FaceCompareService
import time
import asyncio

class ProcessController(BaseController):
    @staticmethod
    async def process(request: ProcessRequest, face_service: FaceRecognitionService):
        start_time = time.time()

        # Liveness
        if request.type == "liveness":
            validate_image = validate_image_ratio(request.filepath)
            if not validate_image["status"]:
                return BaseController.error_response(validate_image["message"])

            embedding_result = face_service.get_embedding(request.filepath)
            if not embedding_result["status"]:
                return BaseController.error_response(embedding_result["message"], 400, embedding_result["data"])

            return BaseController.success_response("Liveness check successful")

        # Compare nor Liveness + Compare 
        elif request.type in ["compare", "both"]:
            # validate_image_1 = validate_image_ratio(request.filepath_1)
            # validate_image_2 = validate_image_ratio(request.filepath_2)

            # if not validate_image_1["status"]:
                # return BaseController.error_response(validate_image_1["message"])
            # if not validate_image_2["status"]:
                # return BaseController.error_response(validate_image_2["message"])

            # embedding_result_1 = face_service.get_embedding(request.filepath_1)
            # embedding_result_2 = face_service.get_embedding(request.filepath_2)

            embedding_result_1, embedding_result_2 = await asyncio.gather(
                face_service.get_embedding(request.filepath_1),
                face_service.get_embedding(request.filepath_2)
            )

            if not embedding_result_1["status"]:
                return BaseController.error_response(f"Error face 1: {embedding_result_1['message']}", 400, embedding_result_1['data'])

            if not embedding_result_2["status"]:
                return BaseController.error_response(f"Error face 2: {embedding_result_2['message']}", 400, embedding_result_2['data'])

            embd_1 = embedding_result_1["data"]['embd']
            embd_2 = embedding_result_2["data"]['embd']
            compare = FaceCompareService.process(embd_1, embd_2)

            data = compare["data"]
            data["conf_1"] = embedding_result_1["data"]["conf"]
            data["conf_2"] = embedding_result_2["data"]["conf"]
            data["tilt_1"] = embedding_result_1["data"]["tilt"]
            data["tilt_2"] = embedding_result_2["data"]["tilt"]
            data["exec_time"] = f"{round(time.time() - start_time, 2)}s" 

            return BaseController.success_response(compare["message"], data)

        return BaseController.success_response("ok")
