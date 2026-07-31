"""
FastAPI application for Kigali Noise Inspection RL Agent.
"""

from fastapi import FastAPI

from api.routes import router



app = FastAPI(

    title="Kigali Noise Inspection RL API",

    description="API for interacting with trained reinforcement learning agents",

    version="1.0.0"

)



app.include_router(
    router
)



@app.get("/")
def home():

    return {

        "project":
            "Kigali Urban Noise Inspection RL",

        "status":
            "running"

    }