# 🧙🔤 Text Wizard

this project is a fun weekend project which  mainly focus on NLP visualization. Thanks to [Sajjad](https://github.com/sajjjadayobi) for his data augmentation technique (Back translation).

In this project We used some amazing tools to experience something new. Now Let's see What we have: 

NOTE : IF YOU ARE NOT USING LINUX MAKE SURE TO CHANGE ALL / to \\\\

### Installation 🔧

First I recommend you use virtual env. then install the libs.
to create a virtual env : *python3 -m venv $HOME/tmp/text_wizard_venv/*

to install all dependencies just use 

> `pip3 install requirements.txt`

but if you are curios about libraries this is the list :

- fast api
- seaborn
- arabic_reshaper
- python_bidi
- scikit-learn
- python-multipart
- mtranslate

## Doc for using features💡

#### first run the code

after installing all the libs then you should start it on your local host , So open your terminal (make sure your Virtual env is ON) then : 

`uvicorn main:app --reload`



Yfor more informative data check the swagger which is created by fast api . more info (https://fastapi.tiangolo.com/)

# Back Translation Api 🌐

What is back translate?

Back translation, also called reverse translation, is **the process of re-translating content from the target language back to its source language** in literal terms. ... A linguist translates the original source text into the new language.

**THIS IS VERY USEFUL TECHNIQUE IN NLP FOR DATA AUGMENTATION**

now you can send **Post** requests to http://127.0.0.1:8000/back_translate route. But make sure your request has these conditions :

- Your request must be in JSON format
- send your text in text parameter 
- set  a list of Languages in an array. make sure your first and last index must be same and they are in same language as your text is example ['fa', 'en', 'fa'] this will translate your text to English and then bring it back to persian. 

your request may be : 

> {
> 	"text" : "من ندانم که چرا در قفس هیچ سگی کرکس نیست",
> 	"lang_list" : [ "fa", "en", "fa" ]
> 	
> }

and you will have the response : 

> "نمی دانم چرا کرکسی در قفس نیست"

you can set any number of languages in a row and then check your results :). that's fun.





#   N_Gram API 🧮

what is N_Gram : In the fields of computational linguistics *and* probability, an *n*-*gram* is a contiguous sequence of *n* items from a given sample of text or speech.



for this API you can send 3 kind of requests. A CSV dataset by setting column which you want to check the ngram of.

more detail about params is here : 

> data : the data you send to get the ngram from, you can send list of strings, one string, or dataframe 
>     but be aware if you send daataframe you must fill col param too
>
> col_name : name of the column you want to get ngram from
>
> ngram = set a int number for n in n-gram ( OPTIONAL, default is 2 )
>
> output_name : name of the jpg file you will get  ( OPTIONAL default is n_gram_plot )
>
> n_most : how many of most repeated ngrams will be shown.  ( OPTIONAL default is  5)

the plot image will be found in ***plots_images*** Folder. in this format output_name_someuniquestrin.jpg

####  if you send string this is a sample POST REQUEST  to http://127.0.0.1:8000/text_ngram: 

> {
>  "text" : "some text",
>  "ngram" : 1,
>  "n_most": 1,
>  "output_name" : "some_name"
> }

and you get this response for all the other requests too : 

> "/plots_images/n_gram_plot_87a490e4-166f-487d-ab6a-a8296731f76e.jpg"

####  if you send list of strings this is a sample POST REQUEST to http://127.0.0.1:8000/text_list_ngram: 

> {
>  "text_list" : ["some text", "some text2"],
>  "ngram" : 1,
>  "n_most": 1,
>  "output_name" : "some_name"
> }

####  and if you send  dataframe to http://127.0.0.1:8000/csv_ngram

you should send your request in Form . make sure your form has this fields : 

> col_name: str  : this is the name of column
> file: UploadFile   : this is the csv file,
> ngram : int : number for n,
> output_name : str : name for output file,
> n_most : int : how many of most repeated ngrams you want to see



#  Rare Words Api 💬

In this part you can check rare words for a text. array of text or column of  a csv file.



for this API you can send 3 kind of requests. A CSV dataset by setting column which you want to check the ngram of.

more detail about params is here : 

> data : the data you send to get the ngram from, you can send list of strings, one string, or dataframe 
>     but be aware if you send daataframe you must fill col param too
>
> col_name : name of the column you want to get ngram from
>
> from_row : the words are sorted by their occurrence. how many rare words you want to see?  [this row ,to number]
>
> to_row = the words are sorted by their occurrence. how many rare words you want to see? [this row ,to number]
>
> output_name : name of the jpg file you will get  ( OPTIONAL default is n_gram_plot )

the plot image will be found in ***plots_images*** Folder. in this format output_name_someuniquestrin.jpg

####  if you send string this is a sample POST REQUEST  to http://127.0.0.1:8000/text_rareword : 

> {
>  "text" : "some text",
>  "from_row" : 1,
>  "to_row": 3,
>  "output_name" : "some_name"
> }

and you get this response for all the other requests too : 

> "/plots_images/n_gram_plot_87a490e4-166f-487d-ab6a-a8296731f76e.jpg"

####  if you send list of strings this is a sample POST REQUEST to http://127.0.0.1:8000/text_list_rareword

> {
>  "text_list" : ["some text", "some text2"],
>  "from_row" : 1,
>  "to_row": 3,
>  "output_name" : "some_name"
> }

####  and if you send  dataframe to http://127.0.0.1:8000/csv_rareword

you should send your request in Form . make sure your form has this fields : 

> col_name: str  : this is the name of column
> file: UploadFile   : this is the csv file,
>  "from_row" : 1,
>  "to_row": 3,
> output_name : str : name for output file,

## Contribute 🖇️



If you have ideas or you want add something feel free to send pull request.
