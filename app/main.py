from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from predict import predict


class Item(BaseModel):
    inp: list[float, float, float, float]

app = FastAPI()


@app.post("/predict/")
async def prediction(item: Item):
    try:
        predicted = predict(item.inp)
        return {"prediction": str(predicted)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=e.__str__())

