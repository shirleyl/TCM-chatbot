from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# 允许前端网页调用
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化DeepSeek客户端
client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": """You are a professional TCM health preservation assistant, skilled in diet therapy and constitution conditioning.
                Please follow these principles:
                1. Guide users to describe specific symptoms (such as sensitivity to cold/heat, sleep, appetite, etc.).
                2. Provide 1-2 simple dietary suggestions based on the symptoms.
                3. If symptoms were complex or uncertain, it's recommended to make an appointment at a clinic.
                4. Always include a disclaimer: The above advice cannot replace a doctor's diagnosis.
                5. Answer in a gentle and caring tone"""},
                {"role": "user", "content": request.message}
            ],
            temperature=0.7,
        )
        return {"reply": response.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"message": "Chatbot API is running"}