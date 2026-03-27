import json
import os

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

app = FastAPI()

# Absolute path to your JSON file
JSON_FILE_PATH = "/home/aatman/Aatman/Study/Semantic Parsing/konkani-amr/annotations/gemini/amr_outputs_100.json"


@app.get("/api/data")
async def get_data():
    """Reads the JSON file and returns the data to the frontend."""
    if not os.path.exists(JSON_FILE_PATH):
        raise HTTPException(
            status_code=404, detail="JSON file not found at the specified path."
        )

    try:
        with open(JSON_FILE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def serve_ui():
    """Serves the index.html file."""
    return FileResponse("index.html")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
