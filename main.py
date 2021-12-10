from typing import List, Optional
from pydantic import BaseModel
from fastapi import FastAPI, File, Form, UploadFile
import pandas as pd

from features.visualization_features_mohammad import word_cloud, POS_WC

sample_text='''
ایران با نام رسمی جمهوری اسلامی ایران، کشوری در آسیای غربی است. این کشور با ۱٬۶۴۸٬۱۹۵ کیلومتر مربع پهناوری، دومین کشور بزرگ خاورمیانه است. ایران از شمال غرب با ارمنستان و آذربایجان، از شمال با دریای خزر، از شمال شرق با ترکمنستان، از شرق با افغانستان و پاکستان، از جنوب با خلیج فارس و دریای عمان و در غرب با عراق و ترکیه هم‌مرز است. این کشور خاورمیانه‌ای، جایگاه استراتژیکی در منطقهٔ خلیج فارس دارد و تنگهٔ هرمز در جنوب آن، مسیری حیاتی برای انتقال نفت خام است. جمعیت کل استان‌های ایران از ۸۳٫۵ میلیون تن می‌گذرد و تهران، پایتخت و پرجمعیت‌ترین شهر این کشور است. ایران، جامعه‌ای با قومیت و فرهنگ‌های گوناگون دارد و گروه قومی و فرهنگی غالب این کشور، برآمده از فارسی‌زبانان آن است. در کنار آنان، قومیت‌های دیگری، همانند اقوام پرجمعیت آذری و کُرد وجود دارند. قانون اساسی جمهوری اسلامی ایران، اسلام شیعه را دین رسمی ایران اعلام کرده‌است و اکثریت مردم این کشور، پیروان همین مذهب هستند. زبان رسمی این کشور نیز فارسی است.

سرزمین ایران، میزبان تمدن‌های کهنی چون ایلام و جیرفت بوده‌است. نخستین‌بار در سدهٔ هفتم پیش از میلاد، در دوران پادشاهی ماد بود که بخش‌های قابل توجهی از فلات ایران یکپارچه شد. در سدهٔ ششم پ. م، شاهنشاهی هخامنشی توسط کوروش بزرگ بنیان نهاده شد تا ایران یکی از بزرگ‌ترین امپراتوری‌های تاریخ را تشکیل دهد. در سدهٔ چهارم پ. م، اسکندر مقدونی این امپراتوری را پایان داد و ایران به بخشی از ممالک هلنیستی تبدیل شد. پدیداری شاهنشاهی اشکانی در سدهٔ سوم پ. م، بار دیگر این کشور را تحت فرمان یک شاهنشاهی ایرانی قرار داد. در سدهٔ سوم م، شاهنشاهی ساسانی، یک امپراتوری گستردهٔ دیگر، در ایران به قدرت رسید و برای چهار سده بر سرزمینی پهناور حکومت کرد و مزدیسنا به دین غالب آن، تبدیل شد. ایران در این دوران نیز درگیر جنگ‌های مستمر و فرساینده با روم بود که به تضعیف کشور انجامید. در میانه‌های سدهٔ هفتم م، مسلمانان، امپراتوری ساسانی را سرنگون کردند و اسلام را به جای دین‌های ایرانی رواج دادند. از دوران خلافت اسلامی تا سدهٔ سیزدهم، فعالیت‌های ادبی، علمی و هنری ایرانی بار دیگر به شکوفایی رسید و ایرانیان مشارکتی اثرگذار در شکل‌گیری دوران طلایی اسلام داشتند. از سدهٔ نهم م، میان‌دورهٔ ایرانی آغاز شد و نخستین حکومت‌های ایرانی‌تبار پس از اسلام، پدیدار شدند. در سدهٔ دهم م، اقوام ترک به این کشور آمدند و حکومت‌هایی را تشکیل دادند که بر بخش بزرگی از ایران، حکومت می‌کردند. از سدهٔ ۱۳ م، حملهٔ مغول به ایران روی داد که به تشکیل ایلخانان انجامید و پس از آن، امپراتوری تیموری پدیدار شد.

با پدیداری صفویان، رونق اقتصادی و پایداری مرزها نمود بیشتری یافت و ایران پس از حدود ۹ سده، تحت یک حکومت مستقل بومی، متحد شد و مذهب آن به شیعه تغییر یافت. پس از سرنگونی صفویان، دودمان‌های افشاریان و زندیان به ترتیب بر ایران، فرمان راندند. در دوران قاجاریان، جنگ‌هایی با روسیه انجام شد که سرزمین‌های قابل توجهی را از این کشور جدا کرد. در دههٔ ۱۲۸۰، جنبش مشروطهٔ ایران قدرت گرفت و قانون اساسی مشروطه را بر این کشور، حاکم کرد. در سال ۱۳۰۴، شاهنشاهی پهلوی توسط رضاشاه بنیان نهاده شد؛ این دوران با اصلاحات گسترده و ایجاد زیرساخت نوین برای ایران، همراه شد. در دوران محمدرضا پهلوی نیز اصلاحات ادامه یافت و ایران، رشد اقتصادی سریعی را تجربه کرد؛ در این دوران، صنعت نفت ایران، ملی شد. سپس نارضایتی‌ها افزایش یافت تا با انقلاب ۱۳۵۷ به رهبری روح‌الله خمینی، این کشور، تحت نظام جمهوری اسلامی اداره شود. از سال ۱۳۵۹ تا ۱۳۶۷ نیز این کشور درگیر جنگی گسترده با عراق بود.

ایران کنونی، یک جمهوری اسلامی با بخش قانون‌گذار است و این نظام ترکیبی، تحت نظر رهبر آن، سید علی خامنه‌ای قرار دارد. ایران از اعضای مؤسس سازمان ملل متحد، سازمان همکاری اقتصادی، سازمان همکاری اسلامی و اوپک است و از قدرت‌های منطقه‌ای شمرده می‌شود. ایران، زیرساخت قابل توجهی در بخش‌های خدماتی، صنعتی و کشاورزی دارد که اقتصاد این کشور را توانمند می‌سازند اما این اقتصاد هنوز بر فروش نفت و گاز متکی است و از فساد مالی رنج می‌برد. منابع طبیعی ایران، قابل توجه هستند و در میان اعضای اوپک، ایران سومین دارندهٔ ذخایر بزرگ اثبات شدهٔ نفت است. میراث فرهنگی این کشور، غنی است و فهرست میراث جهانی یونسکو در ایران از ۲۵ مورد تشکیل می‌شود. 
'''

MAIN_URL = ""
class wc_item(BaseModel):
    
    text : str= sample_text
    remove_stop :Optional[bool]= True
    include_numbers :Optional[bool]= False
    output_name :Optional[str]='word_cloud'
    pos : str = 'v'
    

class wc_list(BaseModel):
    text_list: List[str] = [sample_text]
    remove_stop :Optional[bool]= True
    include_numbers :Optional[bool]= False
    output_name :Optional[str]='word_cloud'
    pos : str = 'v'



app = FastAPI()

#  ============= WordCloud Part =============
@app.post("/text_wordcloud")
async def plot_wordcloud_for_text(item: wc_item):
    image_url = word_cloud(item.text, remove_stop=item.remove_stop, output_name=item.output_name, include_numbers=item.include_numbers)
    return MAIN_URL + image_url

@app.post("/text_list_wordcloud")
async def plot_wordcloud_for_list(item: wc_list):
    image_url = word_cloud(item.text_list, remove_stop=item.remove_stop, output_name=item.output_name, include_numbers=item.include_numbers)
    return MAIN_URL + image_url

@app.post("/csv_wordcloud")
async def plot_wordcloud_for_csv(col_name: str = Form(...) ,
    file: UploadFile = File(...),
    remove_stop :bool= Form(...) ,
    include_numbers :bool= Form(...) ,
    output_name :str=Form(...)):
    if file.filename.endswith('.csv'):
        df = pd.read_csv(file.file)
        image_url = word_cloud(df,col=col_name, remove_stop=remove_stop, output_name=output_name, include_numbers=include_numbers)
        return MAIN_URL + image_url
    else:
        return "Please send a csv file format"

#  ============= Part Of Speech WordCloud Part =============
@app.post("/text_pos_wordcloud/")
async def plot_pos_wordcloud_for_text(item: wc_item):
    image_url = POS_WC(item.text,item.pos, remove_stop=item.remove_stop, output_name=item.output_name, include_numbers=item.include_numbers)
    return MAIN_URL + image_url

@app.post("/text_list_pos_wordcloud")
async def plot_pos_wordcloud_for_list(item: wc_list):
    image_url = POS_WC(item.text_list,item.pos, remove_stop=item.remove_stop, output_name=item.output_name, include_numbers=item.include_numbers)
    return MAIN_URL + image_url

@app.post("/csv_pos_wordcloud")
async def plot_pos_wordcloud_for_csv(col_name: str = Form(...) ,
    file: UploadFile = File(...),
    pos:str= Form(...),
    remove_stop :bool= Form(...) ,
    include_numbers :bool= Form(...) ,
    output_name :str=Form(...)):
    if file.filename.endswith('.csv'):
        df = pd.read_csv(file.file)
        image_url = POS_WC(df,pos,col=col_name, remove_stop=remove_stop, output_name=output_name, include_numbers=include_numbers)
        return MAIN_URL + image_url
    else:
        return "Please send a csv file format"
