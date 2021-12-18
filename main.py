from typing import List, Optional
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi import FastAPI, File, Form, UploadFile
import pandas as pd

from features.visualization_features_masoud import plot_ngram, plot_rare_words
from features.data_augmentation_features_masoud import back_translate

MAIN_URL = ""
class Item(BaseModel):
    text: str = 'HICHI'
    ngram : Optional[int] = 2
    output_name : Optional[str] ='n_gram_plot'
    n_most : Optional[int] =  5
    from_row : Optional[int] = 0
    to_row : Optional[int] = 5

class Text_list(BaseModel):
    text_list: List[str] = []
    ngram : Optional[int] = 2
    output_name : Optional[str] ='n_gram_plot'
    n_most : Optional[int] =  5
    from_row : Optional[int] = 0
    to_row : Optional[int] = 5



app = FastAPI()

@app.post("/text_ngram/")
async def plot_ngram_for_text(item: Item):
    image_url = plot_ngram(item.text, n_gram=item.ngram, output_name=item.output_name, n_most=item.n_most)
    return MAIN_URL + image_url

@app.post("/text_list_ngram")
async def plot_ngram_for_list(item: Text_list):
    image_url = plot_ngram(item.text_list, n_gram=item.ngram, output_name=item.output_name, n_most=item.n_most)
    return MAIN_URL + image_url

@app.post("/csv_ngram")
async def plot_ngram_for_csv(col_name: str = Form(...) ,
    file: UploadFile = File(...),
    ngram : int = Form(...),
    output_name : str =Form(...),
    n_most : int =  Form(...)):
    if file.filename.endswith('.csv'):
        print(ngram)
        df = pd.read_csv(file.file)
        image_url = plot_ngram(df, col = col_name, n_gram=ngram, output_name=output_name, n_most=n_most)
        return MAIN_URL + image_url
    else:
        return "please send a csv file format"

#  ============= Rare words part =============
@app.post("/text_rareword/")
async def plot_rareword_for_text(item: Item):
    print(item)
    image_url = plot_rare_words(item.text, from_row = item.from_row, to_row = item.to_row, output_name = item.output_name)
    return MAIN_URL + image_url

@app.post("/text_list_rareword")
async def plot_rareword_for_list(item: Text_list):
    image_url = plot_rare_words(item.text_list, from_row = item.from_row, to_row = item.to_row, output_name = item.output_name)
    return MAIN_URL + image_url

@app.post("/csv_rareword")
async def plot_rareword_for_csv(col_name: str = Form(...) ,
    file: UploadFile = File(...),
    from_row: int = Form(...),
    output_name : str =Form(...),
    to_row : int =  Form(...)):
    if file.filename.endswith('.csv'):
        df = pd.read_csv(file.file)
        image_url = plot_rare_words(df, col = col_name, output_name=output_name, from_row = from_row, to_row = to_row)
        return MAIN_URL + image_url
    else:
        return "please send a csv file format"


# ============== back translation part ================
#This code is part of SAJJAD AYOUBI source code for his data augmentations techniques
# https://github.com/sajjjadayobi

class Back_translate_data(BaseModel):
    text: str = 'HICHI'
    lang_list: List[str] = ['fa', 'en', 'fa']

@app.post("/back_translate")
async def back_translate_feature(item : Back_translate_data):
    if item.text == "HICHI" : return "please send some string"
    return back_translate(text = item.text, lang_list = item.lang_list)