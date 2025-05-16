from controllers.base_controller import BaseController
from form_requests.process_request import ProcessRequest
from validators.image_validator import validate_image_ratio
from services.face_liveness_service import FaceLiveness
from services.face_recognition_service import FaceRecognitionService
from services.face_compare_service import FaceCompareService
import time
import asyncio
import os

class ProcessController(BaseController):
    @staticmethod
    async def process(request: ProcessRequest, face_service: FaceRecognitionService):
        start_time = time.time()

        # Liveness
        if request.type == "liveness":
            # validate_image = validate_image_ratio(request.filepath)
            # if not validate_image["status"]:
            #     return BaseController.error_response(validate_image["message"])

            embedding_result = await face_service.get_embedding(request.filepath)
            if not embedding_result["status"]:
                return BaseController.error_response(embedding_result["message"], 400, embedding_result["data"])

            face_liveness = FaceLiveness()
            liveness = face_liveness.check(embedding_result["data"]["img"], embedding_result["data"]["bbox"])
            data = {
                "result": liveness["data"]["result"],
                "score": liveness["data"]["score"],
                "exec_time": f"{round(time.time() - start_time, 2)}s" 
            }

            return BaseController.success_response(liveness["message"], data)


        # Compare nor Liveness + Compare 
        elif request.type in ["compare", "both"]:
            # validate_image_1 = validate_image_ratio(request.filepath_1)
            # validate_image_2 = validate_image_ratio(request.filepath_2)

            # if not validate_image_1["status"]:
                # return BaseController.error_response(validate_image_1["message"])
            # if not validate_image_2["status"]:
                # return BaseController.error_response(validate_image_2["message"])

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
            
            data = {}
            
            if request.type == 'both':
                face_liveness = FaceLiveness()
                liveness_1, liveness_2, compare = await asyncio.gather(
                    face_liveness.check(embedding_result_1["data"]["img"], embedding_result_1["data"]["bbox"]),
                    face_liveness.check(embedding_result_2["data"]["img"], embedding_result_2["data"]["bbox"]),
                    FaceCompareService.process(embd_1, embd_2)
                )

                data["liveness_image_1"] = liveness_1["data"]["result"]
                data["liveness_image_2"] = liveness_2["data"]["result"]

            else:    
                compare = FaceCompareService.process(embd_1, embd_2)

            data["tolerance"]   = os.getenv("FACE_SIMILARITY_THRESHOLD")
            data["similarity"]  = compare["data"]["similarity"]
            data["result"]      = compare["data"]["result"]
            data["conf_1"]      = embedding_result_1["data"]["conf"]
            data["conf_2"]      = embedding_result_2["data"]["conf"]
            # data["tilt_1"] = embedding_result_1["data"]["tilt"]
            # data["tilt_2"] = embedding_result_2["data"]["tilt"]
            data["exec_time"]   = f"{round(time.time() - start_time, 2)}s" 

            return BaseController.success_response(compare["message"], data)

        return BaseController.success_response("ok")
