from fastapi import APIRouter, HTTPException
from app.personalization import UserProfile, PersonalizationModel
from app.image_generation import ImageGenerator
from pydantic import BaseModel

router = APIRouter()
user_profiles = {}
personalization_model = PersonalizationModel()
image_generator = ImageGenerator()

class ImageRequest(BaseModel):
    prompt: str

@router.post("/update-preferences")
def update_preferences(user_id: str, category: str, key: str, value: str):
    if user_id not in user_profiles:
        user_profiles[user_id] = UserProfile(user_id)
    user_profiles[user_id].update_preferences(category, key, value)
    return {"message": "Preferences updated successfully."}

@router.get("/get-suggestion")
def get_suggestion(user_id: str):
    if user_id not in user_profiles:
        raise HTTPException(status_code=404, detail="User not found.")
    # Simulação de interação para o modelo
    interaction = np.array([1, 0, 1])  # Exemplo
    suggestion = personalization_model.predict(interaction)
    return {"suggestion": suggestion}




@router.post("/generate-image")
def generate_image(request: ImageRequest):
    prompt = request.prompt
    image_path = f"generated_images/{prompt.replace(' ', '_')}.png"
    image_generator.generate_image(prompt, image_path)
    return {"message": "Image generated successfully.", "image_path": image_path}