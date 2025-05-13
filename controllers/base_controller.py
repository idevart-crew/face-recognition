from fastapi.responses import JSONResponse

class BaseController:
    @staticmethod
    def success_response(message: str, data=None):
        return JSONResponse(
            status_code=200,
            content={"status": 200, "message": message, "data": data},
        )

    @staticmethod
    def error_response(message: str, status_code=400, data=None):
        return JSONResponse(
            status_code=status_code,
            content={"status": status_code, "message": message, "data": data},
        )
