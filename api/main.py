from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
from api.tool import retrieval

app = FastAPI()

class Parameters(BaseModel):
    index : str
    query : str

@app.get("/")
async def health_check():
    return {"status":"ok"}

@app.post('/inference')
async def get_response(parameters : Parameters):
    args = parameters.model_dump()
    respone = retrieval(args["index"], args["query"])
    return {"inference": respone}

if __name__ == "__main__":
    uvicorn.run(app)

