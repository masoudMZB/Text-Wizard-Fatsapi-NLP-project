from mtranslate import translate

def back_translate(text ,lang_list = ["fa", "en", "fa"]):
  """
  this function will plot most reapeted ngrams for you
  parameters : 
    text : send your text in your source language. default is persian

    lang_list : send a list of languages BUT BECAREFUL first index and last index should be same
    make sure you write langs in proper format for example English is en

  """

  augmented_text = text
  for i, lang in enumerate(lang_list):
    if i+1 < len(lang_list) :
      augmented_text = translate(augmented_text, from_language=lang_list[i], to_language=lang_list[i+1])
      print(augmented_text)
  return augmented_text