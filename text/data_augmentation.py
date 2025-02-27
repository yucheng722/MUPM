import json
import nltk
import random
import pandas as pd
import re

def reduce_text(background_sentences):
 # 读取 Excel
  sentences = re.split(r'(?<=[.!?])\s+', str(background_sentences))
  selected = random.sample(sentences, int(len(sentences)/3))
  return " ".join(selected)

def sample_text_augmentation(text,Question,background_dir,eid,time):
    I = random.sample(range(30), time)
    for i in I:
        dir = background_dir + '{}.csv'.format(i+1)
        df_bk = pd.read_csv(dir)
        item = df_bk.loc[df_bk['eid'] == eid]
        background_sentences = item['background'].values[0]
        #background_sentences_reduced = reduce_text(background_sentences)

        match = re.findall(r'(\d+(?:\.\d+)?)\s+years', text)
        last_year_value = match[-1] if match else None
        #prompt_path = r"D:\NewPythonProject\Yucheng\text\multi_choice_prompt.json"
        prompt_path = '/SAN/medic/candi_tyc/Yucheng/text/multi_choice_prompt.json'
        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompt_data = json.load(f)
        prompt_data = list(prompt_data.values())[0]
        prompt_sentences = random.choice(prompt_data).format(year = last_year_value)

        #split_text = re.split(r'\s+(?=Options:)', text, maxsplit=1)
        split_text = re.split(r'\s*(?=Options:)', text, maxsplit=1)

        option_sentences = split_text[1] if len(split_text) > 1 else ""
        Question.append(background_sentences + prompt_sentences +
                        option_sentences )

    return Question

#def sample_image_augmentation(image,image_path_list,IMG_Path_dir,eid):
#    for i in range(11):
#        dir = IMG_Path_dir + '{}.csv'.format(i+1)
#        df_bk = pd.read_csv(dir)
#        item = df_bk.loc[df_bk['eid'] == eid]

#import nltk
#nltk.download('punkt_tab',download_dir=r"C:\Users\Lenovo\nltk_data")
#text_dir = r"D:\NewPythonProject\Yucheng\evaluation.xlsx"
#df = pd.read_excel(text_dir)
#Que = df['question']
#que = []
#for q in Que:
#    sentences = nltk.sent_tokenize(q)
#    que.append(sentences[-4:-1])
#df['que'] = que
#df.to_excel(r"D:\NewPythonProject\Yucheng\evaluation_reduced.xlsx")


#def keep_last_4_sentences(text):
#    split_text = re.split(r'\s+(?=Options:)', text, maxsplit=1)
#    background_sentences_reduced = reduce_text(split_text[0])

#    match = re.findall(r'(\d+(?:\.\d+)?)\s+years', text)
#    last_year_value = match[-1] if match else None
    # prompt_path = r"D:\NewPythonProject\Yucheng\text\multi_choice_prompt.json"
#    prompt_path = r"D:\NewPythonProject\Yucheng\text\multi_choice_prompt.json"
#    with open(prompt_path, 'r', encoding='utf-8') as f:
#        prompt_data = json.load(f)
#    prompt_data = list(prompt_data.values())[0]
#    prompt_sentences = random.choice(prompt_data).format(year=last_year_value)

#   option_sentences = split_text[1] if len(split_text) > 1 else ""
#    full = background_sentences_reduced + prompt_sentences + option_sentences
#    return full # 取最后4句

#df["question"] = df["question"].apply(keep_last_4_sentences)

# 保存回 Excel
#df.to_excel(r"D:\NewPythonProject\Yucheng\evaluation_reduced.xlsx")
