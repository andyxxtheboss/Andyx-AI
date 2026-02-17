import os
import google.generativeai as genai
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import uvicorn

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Folosim 1.5-flash pentru stabilitate maxima pe Free Tier
model = genai.GenerativeModel("gemini-1.5-flash")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Istoric conversație
chat_history = []

class ChatRequest(BaseModel):
    message: str = ""
    image: str | None = None  # base64

@app.post("/chat")
async def chat_with_andyx(request: ChatRequest):
    try:
        contents = []

        # 1️⃣ Mesajul utilizatorului
        user_parts = []
        if request.message.strip():
            user_parts.append({"text": request.message.strip()})
        
        if request.image and len(request.image) > 100:
            user_parts.append({
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": request.image
                }
            })
        
        # Dacă nu avem nici text nici imagine, punem un default
        if not user_parts:
            user_parts.append({"text": "Salut!"})

        # Adăugăm mesajul nou în context
        current_turn = {
            "role": "user",
            "parts": user_parts
        }

        # 2️⃣ Construim contextul: Istoric + Mesaj nou
        full_context = []
        for turn in chat_history[-10:]: # Ultimele 10 replici
            full_context.append(turn)
        full_context.append(current_turn)

        # 3️⃣ Apel Gemini cu tratare eroare de Quota
        try:
            response = model.generate_content(full_context)
        except Exception as gem_err:
            err_msg = str(gem_err)
            if "429" in err_msg or "quota" in err_msg.lower():
                return {"reply": "⚠️ Andyx are nevoie de o mică pauză (limită de mesaje atinsă). Revino în un minut!"}
            raise gem_err

        # 4️⃣ Extragem răspunsul
        model_reply = ""
        if response and response.text:
            model_reply = response.text.strip()
        
        if not model_reply:
            model_reply = "Andyx a primit mesajul, dar nu a putut genera un răspuns text."

        # 5️⃣ Salvăm în istoric ambele părți pentru context viitor
        chat_history.append(current_turn)
        chat_history.append({
            "role": "model",
            "parts": [{"text": model_reply}]
        })

        # Menținem istoricul scurt pentru a nu depăși limitele de memorie
        if len(chat_history) > 20:
            chat_history.pop(0)
            chat_history.pop(0)

        return {"reply": model_reply}

    except Exception as e:
        print(f"EROARE SERVER: {e}")
        return {"reply": f"Eroare tehnică: {str(e)}"}

# --- MODIFICAREA PENTRU DEPLOY ---
if __name__ == "__main__":
    # Citim portul setat de serverul de hosting (Render/Railway), implicit 8000
    port = int(os.environ.get("PORT", 8000))
    print(f"--- Serverul Andyx este ONLINE pe portul {port} ---")
    uvicorn.run(app, host="0.0.0.0", port=port)