import os.path
import pandas as pd
import gc
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
from image.image_preprocessing import get_npy
device = torch.device('cuda') # 'cpu', 'cuda'
dtype = torch.bfloat16 # or bfloat16, float16, float32
from text.data_augmentation import sample_text_augmentation

torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)



def append_to_excel(file_path, data_frame, sheet_name='Sheet1'):
    try:
        with pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
            data_frame.to_excel(writer, sheet_name=sheet_name, index=False, header=False, startrow=writer.sheets[sheet_name].max_row)
    except FileNotFoundError:
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            data_frame.to_excel(writer, sheet_name=sheet_name, index=False)


def get_text_info(text_dir,time):
    df = pd.read_excel(text_dir)
    if time == 1:
      filtered_df = df[~df['answer'].str.contains(r'No|none of above', case=False, na=False)]
      sampled_data = filtered_df.to_dict('records')
      sample_size = max(1, int(len(sampled_data) * 1.0))  #Percentage of sampled data
      sampled_data = random.sample(sampled_data, sample_size)
    else:
        sampled_data = df.to_dict('records')
    return sampled_data

def get_pre(model,tokenizer,proj_out_num,sampled_data,
            item_dir_list,name_dir_list,background_dir,output_path,
            save_result_dir,n):
    i = 0
    for item in sampled_data:
      for time in [1,2,3]:
        if time == 1: # The first time: Image-only
          Image_aug, Text_aug = True, False
        if time == 2:# The second time: Text-only
          Image_aug, Text_aug = False, True
        if time == 3:# The third time: Image-Text
          Image_aug, Text_aug = True, True
        name = item['eid']
        image_path_list = []
        Question = []
        if Image_aug == True:
          image_path_list.append(get_npy(item_dir_list, output_path,name,
                                         name_dir_list,False,0))
          for time in np.arange(augment_times): #
              image_path_list.append(get_npy(item_dir_list, output_path, name,
                                             name_dir_list,True,
                                             time))
        else: image_path_list.append(get_npy(item_dir_list,output_path, name,
                                         name_dir_list,False,0))


        if Text_aug == True:
            Question.append(item['question'])
            Question = sample_text_augmentation(item['question'],Question,
                                    background_dir,eid = name,time=20)
        else:
          Question.append(item['question'])

        image_tokens = "<im_patch>" * proj_out_num


        if Image_aug == True and Text_aug == False:
          time = 0
          for image_path in image_path_list:
            image_np = np.load(image_path+'.npy')
            image_pt = torch.from_numpy(image_np).unsqueeze(0).to(dtype=dtype, device=device)
            input_txt =  (f"The sax MRI images are as follows: {image_tokens}, it has 32 time dimensions. "
                        f"{Question[0]} You can select more than 1 correct answers")
            input_id = tokenizer(input_txt, return_tensors="pt")['input_ids'].to(device=device)

            with torch.no_grad():
              generation = model.generate(image_pt, input_id,max_new_tokens=64, do_sample=True, top_p=0.9, temperature=1.0)
              outputs = model(image_pt, input_id)

            generated_texts = tokenizer.batch_decode(generation, skip_special_tokens=True)
            print('question', Question[0])
            print('generated_texts', generated_texts)
            item['pre_{}'.format(time)] = generated_texts
            print(time)
            time += 1


        if Text_aug == True and Image_aug == False:
            time = n + 1
            for question in Question:
                image_np = np.load(image_path_list[0] + '.npy')
                image_pt = torch.from_numpy(image_np).unsqueeze(0).to(dtype=dtype, device=device)
                input_txt = (f"The sax MRI images are as follows: {image_tokens}, it has 32 time dimensions. "
                             f"{question} You can select more than 1 correct answers")
                input_id = tokenizer(input_txt, return_tensors="pt")['input_ids'].to(device=device)
                with torch.no_grad():
                    generation = model.generate(image_pt, input_id, max_new_tokens=64, do_sample=True, top_p=0.9,
                                                temperature=1.0)
                    outputs = model(image_pt, input_id)


                generated_texts = tokenizer.batch_decode(generation, skip_special_tokens=True)
                print('question', question)
                print('generated_texts', generated_texts)
                item['pre_{}'.format(time)] = generated_texts
                print(time)
                time += 1


        if Image_aug == True and Text_aug == True:
            time = (n + 1) * 2
            for image_path,question in zip(image_path_list,Question):
                image_np = np.load(image_path + '.npy')
                image_pt = torch.from_numpy(image_np).unsqueeze(0).to(dtype=dtype, device=device)
                input_txt = (f"The sax MRI images are as follows: {image_tokens}, it has 32 time dimensions. "
                             f"{question} You can select more than 1 correct answers")
                input_id = tokenizer(input_txt, return_tensors="pt")['input_ids'].to(device=device)
                with torch.no_grad():
                    generation = model.generate(image_pt, input_id, max_new_tokens=64, do_sample=True, top_p=0.9,
                                                temperature=1.0)
                    outputs = model(image_pt, input_id)

                generated_texts = tokenizer.batch_decode(generation, skip_special_tokens=True)
                print('question', question)
                print('generated_texts', generated_texts)
                item['pre_{}'.format(time)] = generated_texts
                print(time)
                time += 1

            nn= pd.concat([pd.DataFrame(item)], axis=1)
            append_to_excel(save_result_dir, nn)

        # Delete variables to free up memory
        del image_np
        del image_pt
        del generation,outputs
        del generated_texts

        # Clear CUDA memory (if using GPU)
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        # Optional: force garbage collection
        gc.collect()

        i += 1
        print(i)

    return sampled_data


#model
model_name_or_path = 'GoodBaiBai88/M3D-LaMed-Phi-3-4B'
proj_out_num = 256
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype=dtype,
    device_map='auto',
    trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path,
    model_max_length=1024,
    padding_side="right",
    use_fast=False,
    trust_remote_code=True
)

print('Model loading finished')

#image
IMG_list_dir = ".../image/img_dir_text.xlsx"
IMG_output_path = os.path.join('.../image/', 'npy_data')
print('Image process finished')

#text
background_dir = ".../text/background"
save_result_dir = ".../results.xlsx"
df2 = pd.read_excel(IMG_list_dir)
item_dir_list = df2['Item'].tolist() #image item dir
name_dir_list = df2['Name'].tolist() #image name
text_dir = '.../text/evaluation.xlsx'

sampled_data = get_text_info(text_dir,1)

sampled_data = get_pre(model, tokenizer, proj_out_num, sampled_data,
                        item_dir_list, name_dir_list,
                        background_dir, IMG_output_path,save_result_dir,
                       n = 20)
    
print(sampled_data.__len__())
