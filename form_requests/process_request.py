from form_requests.base_request import BaseRequest
from pydantic import Field, model_validator
from typing import Optional

class ProcessRequest(BaseRequest):
    type: str = Field(..., description="Type must be 'compare', 'liveness', or 'both'")
    filepath: Optional[str] = None
    filepath_1: Optional[str] = None
    filepath_2: Optional[str] = None

    @model_validator(mode="after")
    def check_required_fields(self):
        """Validasi field yang wajib berdasarkan type"""
        if self.type == "liveness":
            if not self.filepath:
                raise ValueError("filepath is required for type 'liveness'")
        
        if self.type in ["compare", "both"]:
            if not self.filepath_1:
                raise ValueError("filepath_1 is required for type '{}'".format(self.type))
            if not self.filepath_2:
                raise ValueError("filepath_2 is required for type '{}'".format(self.type))

        return self
