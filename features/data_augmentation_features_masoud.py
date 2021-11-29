from mtranslate import translate

def back_translate(text ,lang_list = ["fa", "en", "fa"]):
  augmented_text = text
  for i, lang in enumerate(lang_list):
    if i+1 < len(lang_list) :
      augmented_text = translate(augmented_text, from_language=lang_list[i], to_language=lang_list[i+1])
      print(augmented_text)
  return augmented_text