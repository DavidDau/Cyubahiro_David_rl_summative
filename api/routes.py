"""
API routes for Kigali Noise Inspection RL Agent.
"""

from fastapi import APIRouter

from pydantic import BaseModel

from stable_baselines3 import PPO

from environment.custom_env import NoiseInspectionEnv



router = APIRouter()



MODEL_PATH = "models/ppo/noise_ppo.zip"



model = None



env = NoiseInspectionEnv()



def load_model():

    global model

    if model is None:

        model = PPO.load(
            MODEL_PATH
        )

    return model



class ActionRequest(BaseModel):

    state: list



class ActionResponse(BaseModel):

    action: int

    zone: int

    confidence: float



@router.get("/health")
def health_check():

    return {

        "status":
            "API is running"

    }



@router.post(
    "/predict",
    response_model=ActionResponse
)
def predict_action(
    request: ActionRequest
):

    agent = load_model()



    state = request.state



    action, _ = agent.predict(

        state,

        deterministic=True

    )



    if hasattr(action, "item"):

        action = action.item()



    return {

        "action":
            int(action),

        "zone":
            env.current_zone,

        "confidence":
            1.0

    }



@router.post("/reset")
def reset_environment():

    observation, _ = env.reset()


    return {

        "initial_state":
            observation.tolist()

    }