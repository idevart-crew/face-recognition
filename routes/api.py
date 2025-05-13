from fastapi import APIRouter, Depends
from controllers.base_controller import BaseController
from controllers.process_controller import ProcessController
from form_requests.process_request import ProcessRequest
from services.face_recognition_service import get_face_service

router = APIRouter()

@router.get('/')
def index():
    return BaseController.success_response('Server is up!')

@router.post('/process')
async def process_request(request: ProcessRequest, face_service=Depends(get_face_service)):
    return await ProcessController.process(request, face_service)
