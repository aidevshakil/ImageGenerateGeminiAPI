import base64
import os
import mimetypes
import logging
from google import genai
from google.genai import types
from app.utils.env_loader import load_environment
from fastapi import HTTPException

load_environment()

logger = logging.getLogger("imageboot")
logging.basicConfig(level=logging.DEBUG)

def categorize_prompt(prompt):
    """
    Automatically categorize the user's prompt
    
    Args:
        prompt (str): User's image generation request
    
    Returns:
        str: Detected image category
    """
    categories = {
        "Funny": ["funny", "humor", "joke", "comedy", "laugh"],
        "Educational": ["diagram", "explain", "science", "learn", "concept"],
        "Geographical": ["map", "landscape", "location", "terrain", "geography"],
        "Animation": ["cartoon", "anime", "animated", "character", "style"],
        "Science Fiction": ["scifi", "future", "space", "alien", "technology"],
        "Horror": ["scary", "horror", "dark", "creepy", "monster"]
    }
    
    # Convert prompt to lowercase for easier matching
    lower_prompt = prompt.lower()
    
    # Check for category keywords
    for category, keywords in categories.items():
        if any(keyword in lower_prompt for keyword in keywords):
            return category
    
    # Default category if no specific match
    return "General"

def save_binary_file(file_name, data):
    f = open(file_name, "wb")
    f.write(data)
    f.close()

def generate(user_prompt):
    logger.debug(f"Generating image for prompt: {user_prompt}")
    client = genai.Client(
        api_key=os.getenv("GEMINI_API_KEY"),
    )
    category = categorize_prompt(user_prompt)

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=f"""
                Generate a creative and detailed image based on this prompt:
                {user_prompt}

                Image Category: {category}

                Guidelines for Image Generation:
                - Create a high-quality, creative image
                - Accurately represent the user's request
                - Maintain appropriate and engaging visual style
                - Focus on the specific {category} category characteristics

                For this image, provide:
                1. A detailed description of the generated image
                2. Three key visual elements in the image
                3. The artistic style or mood of the image
                4. How the image reflects the {category} category

                Generate the image and then describe it comprehensively.
                """),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=1,
        top_p=0.95,
        top_k=40,
        max_output_tokens=8192,
        response_modalities=[
            "image",
            "text",
        ],
        safety_settings=[
            types.SafetySetting(
                category="HARM_CATEGORY_HARASSMENT",
                threshold="BLOCK_NONE",  # Block none
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_HATE_SPEECH",
                threshold="BLOCK_NONE",  # Block none
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                threshold="BLOCK_NONE",  # Block none
            ),
        ],
        response_mime_type="text/plain",
    )
    MODEL_NAME = "gemini-2.0-flash-exp-image-generation"

    try:
        for chunk in client.models.generate_content_stream(
            model=MODEL_NAME, contents=contents, config=generate_content_config
        ):
            if chunk.candidates and chunk.candidates[0].content.parts:
                part = chunk.candidates[0].content.parts[0]
                if part.inline_data:
                    # Save image to static/images directory
                    file_name = f"generated_image_{hash(user_prompt)}"
                    file_extension = mimetypes.guess_extension(part.inline_data.mime_type) or ".png"
                    save_path = f"static/images/{file_name}{file_extension}"
                    save_binary_file(save_path, part.inline_data.data)

                    return {"image_url": f"/static/images/{file_name}{file_extension}", "category": category}
    
    except Exception as e:
        print(f"Error generating image: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate image.")

    raise HTTPException(status_code=500, detail="No valid response received from the API.")


        
if __name__ == "__main__":
    generate("Test prompt")
    
