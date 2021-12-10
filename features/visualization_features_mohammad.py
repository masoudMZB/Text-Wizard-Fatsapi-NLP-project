from wordcloud_fa import WordCloudFa
from os.path import join
import os
import pandas as pd
from hazm import POSTagger,word_tokenize
from datetime import datetime


def word_cloud(data,col='not_set_yet',remove_stop=True,include_numbers=False,output_name='word_cloud'):
    '''
    This function take data and make wordcloud of it. It save wordcloud in img_url

    Parameters
    ----------
    data : list/str/pd.DataFrame
        List of str or pd.DataFrame.
    col : str/int, optional
        Neccessary if data is pd.DataFrame
        It should be the name of column contain data.
        The default is 'not_set_yet'.
    remove_stop : Bool, optional
        True if you want to remove stopword, False otherwise. The default is True.
    include_numbers : Bool, optional
        True if you want to include numbers in wordcloud, False otherwise. The default is False.
    output_name : str, optional
        First part of name of final image, It will save in img_url. The default is 'word_cloud'.

    Raises
    ------
    Exception
        If data provided as list, all element of it must be str, otherwise Exception will rais.
    Exception
        If data provided as pd.DataFrame, col must be specified.
    Exception
        If remove_stop set ad True, stopwords.dat must exist in ./resources, otherwise Exception will rais.

    Returns
    -------
    img_url : str
        Path to generated wordcloud image.

    '''
    text=data
    if isinstance(data, list):
        if not all(map(lambda x:isinstance(x, str),data)): raise Exception("All elements of list must be str")
        text=' '.join(data)
    if isinstance(data, pd.DataFrame):
        if col == "not_set_yet" : raise Exception("please set which column you want to check")
        text=' '.join(data[col].tolist())
        
    wc=WordCloudFa(stopwords=set(),persian_normalize=True,include_numbers=include_numbers,background_color='white',no_reshape=True,collocations=False,width=800,height=400)
    if remove_stop:
        if os.path.isfile(join('.','resources','stopwords.dat')):
            wc.add_stop_words_from_file(join('.','resources','stopwords.dat'))
        else :
            raise Exception("Please download stopwords.dat from hazm and put it on resources folder in root path. Read README.md for more information")
    wc.generate(text)
    if not os.path.exists(join('.','plots_images')):
        os.makedirs(join('.','plots_images'))
    img_name = f'{output_name}_{datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}.jpg'
    img_url = join('.','plots_images',img_name)
    wc.to_image().save(img_url,dpi=(300,300))
    
    return img_url
    
def POS_WC(data,POS,col='not_set_yet',remove_stop=False,include_numbers=False,output_name='POS_word_cloud'):
    '''
    This function take data and make wordcloud of its Verbs/Adjectives. It save wordcloud in img_url

    Parameters
    ----------
    data : list/str/pd.DataFrame
        List of str or pd.DataFrame.
    POS : str
        adj for Adjectives.
        v for Verbs
    col : str/int, optional
        Neccessary if data is pd.DataFrame
        It should be the name of column contain data.
        The default is 'not_set_yet'.
    remove_stop : Bool, optional
        True if you want to remove stopword, False otherwise. The default is False.
    include_numbers : Bool, optional
        True if you want to include numbers in wordcloud, False otherwise. The default is False.
    output_name : str, optional
        First part of name of final image, It will save in img_url. The default is 'POS_word_cloud'.

    Raises
    ------
     Exception
        If data provided as pd.DataFrame, col must be specified.
    Exception
        postagger.model must exist in ./resources, otherwise Exception will rais. 
    Exception
        Valid POS must specified, otherwise Exception will raise. Valids are 'adj' and 'v'.
    
    Returns
    -------
    img_url : str
        Path to generated wordcloud image.

    '''
    text=data
    if isinstance(text, list):
        text=' '.join(data)
    elif isinstance(data, pd.DataFrame):
        if col == "not_set_yet" : raise Exception("Please set which column you want to check")
        text=' '.join(data[col].tolist())
        
    if os.path.isfile(join('.','resources','postagger.model')):
        tagger=POSTagger(model=join('.','resources','postagger.model'))
    else:
        raise Exception("Please download tagger model from hazm and put it on resources folder in root path. Read README.md for more information")
    POS_tags=tagger.tag(word_tokenize(text.strip()))
    
    if POS.lower()=='v':
        words_tags=list(filter(lambda x:x[1]=='V',POS_tags))
        words=list(map(lambda x:x[0],words_tags))
        new_text=' '.join(words)
        if output_name=='POS_word_cloud':
            output_name='word_cloud_verb'
    elif POS.lower()=='adj':
        words_tags=list(filter(lambda x:x[1]=='AJe' or x[1]=='AJ',POS_tags))
        words=list(map(lambda x:x[0],words_tags))
        new_text=' '.join(words)
        if output_name=='POS_word_cloud':
            output_name='word_cloud_adjective'
    else:
        raise Exception("Please provide a valid POS tag:\nv for Verbs\t adj for Adjectives")
    
    img_url=word_cloud(new_text,remove_stop=remove_stop,include_numbers=include_numbers,output_name=output_name)
    return img_url
        
