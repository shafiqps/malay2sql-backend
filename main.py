from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import auth, users, malay2sql
from database import create_tables

app = FastAPI(title="MalaySQL Backend")

# Create database tables at startup
create_tables()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Add your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router)
app.include_router(users.router)
app.include_router(malay2sql.router)

@app.get("/")
async def root():
    return {"message": "Welcome to MalaySQL Backend"}