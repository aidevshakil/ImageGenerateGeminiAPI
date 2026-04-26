from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from app.models.request_models import ImageGenerationRequest
from app.services.imageboot import generate
import logging

# Initialize logger
logger = logging.getLogger("image_routes")
logging.basicConfig(level=logging.DEBUG)

image_router = APIRouter()

@image_router.post("/generate")
async def generate_image(request: ImageGenerationRequest):
    """
    API endpoint to generate an image and return it in Base64 format.
    """
    try:
        logger.debug(f"Received prompt: {request.prompt}")
        result = generate(request.prompt)
        logger.debug(f"Result from generate function: {result}")

        # Check for "image_url" in the result
        if "image_url" in result:
            return JSONResponse(
                content={
                    "status": "success",
                    "message": "Image generated successfully",
                    "image_url": result["image_url"],  # Use "image_url" instead of "image"
                    "category": result["category"]
                }
            )
        else:
            logger.error("Image generation failed: No image in result.")
            return JSONResponse(content={"status": "error", "message": "Image generation failed"}, status_code=500)
    except Exception as e:
        logger.exception("Error occurred during image generation.")
        raise HTTPException(status_code=500, detail=str(e))