from pydantic import BaseModel, Field

class ImageGenerationRequest(BaseModel):
    prompt: str = Field(..., min_length=5, max_length=500, 
                        description="Detailed description for image generation")
    
    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "A futuristic city landscape at sunset with flying cars"
            }
        }

