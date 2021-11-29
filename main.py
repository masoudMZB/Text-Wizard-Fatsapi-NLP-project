from typing import List, Optional
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi import FastAPI, File, Form, UploadFile
import pandas as pd

from features.visualization_features_masoud import plot_ngram

MAIN_URL = ""
class Item(BaseModel):
    text: str = 'HICHI'

class Text_list(BaseModel):
    text_list: List[str] = []

app = FastAPI()

@app.post("/text_ngram/")
async def plot_ngram_for_text(item: Item):
    image_url = plot_ngram(item.text)
    return MAIN_URL + image_url

@app.post("/text_list_ngram")
async def plot_ngram_for_list(item: Text_list):
    image_url = plot_ngram(item.text_list)
    return MAIN_URL + image_url

@app.post("/csv_ngram")
async def plot_ngram_for_csv(col_name: str = Form(...) ,file: UploadFile = File(...)):
    if file.filename.endswith('.csv'):
        df = pd.read_csv(file.file)
        image_url = plot_ngram(df, col = col_name)
        return MAIN_URL + image_url
    else:
        return "please send a csv file format"
