from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from routes.api import router
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    messages = []
    
    for error in errors:
        field = ".".join(str(loc) for loc in error["loc"])
        msg = error["msg"]
        messages.append(f"{field}: {msg}")

    return JSONResponse(
        status_code=422,
        content={
            "status": 422,
            "message": " | ".join(messages),
            "data": None
        },
    )

app.include_router(router)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000, reload=True)
