from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers.vector_route import router

app = FastAPI()

# 👇 Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development only. Use specific domains in production.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 👇 Mount your router
app.include_router(router)

@app.get('/')
def testing():
    return {"message": "Backend successfully working"}
