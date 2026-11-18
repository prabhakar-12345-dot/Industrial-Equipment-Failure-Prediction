from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

import pandas as pd
import joblib

app = FastAPI()

model = joblib.load("industrial_failure_model.joblib")

templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )


@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    air_temperature: float = Form(...),
    process_temperature: float = Form(...),
    rotational_speed: float = Form(...),
    torque: float = Form(...),
    tool_wear: float = Form(...),
    machine_type: str = Form(...)
):

    type_l = 1 if machine_type == "L" else 0
    type_m = 1 if machine_type == "M" else 0

    input_data = pd.DataFrame([{
        "Air temperature [K]": air_temperature,
        "Process temperature [K]": process_temperature,
        "Rotational speed [rpm]": rotational_speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear,
        "TWF": 0,
        "HDF": 0,
        "PWF": 0,
        "OSF": 0,
        "RNF": 0,
        "Type_L": type_l,
        "Type_M": type_m
    }])

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        result = "Machine Failure"
    else:
        result = "No Machine Failure"

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "prediction_text": result
        }
    )