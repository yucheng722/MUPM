import os.path
import pandas as pd
from accelerate import dispatch_model
import gc
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import SimpleITK as sikt
import nibabel as nib
import random
from scipy.ndimage import zoom
from extract import find_data_dir
from collections import Counter
from image.metrics import exact_match_score,precision_recall_f1_score,top_k_accuracy

device = torch.device('cuda') # 'cpu', 'cuda'
dtype = torch.bfloat16 # or bfloat16, float16, float32
from data.affine import (
    BatchImageLabelRandomAffine,
    BatchImageRandomAffine,
)
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

def append_to_json(file_path, data_frame, orient="records", indent=4):
    try:
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                existing_data = json.load(file)
                if not isinstance(existing_data, list):
                    raise ValueError("JSON data is not in list format and data cannot be appended")
        except (FileNotFoundError, json.JSONDecodeError):
            existing_data = []
        new_data = json.loads(data_frame.to_json(orient=orient))
        existing_data.extend(new_data)
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(existing_data, file, indent=indent, ensure_ascii=False)
        print("Json successfully appended")
    except Exception as e:
        print(f"Json error: {e}")

def data_affine(image_np):
    spacing = (1.0, 1.0,10.0)
    image_size = image_np[...].shape
    image_np = image_np[None, None, ...]
    image_np = image_np.astype(np.float32) / 255.
    image = torch.from_numpy(image_np)

    align_corners = False
    max_rotation = (15, 15, 15)  # degrees for yz, xz, xy planes
    max_zoom = (0.2, 0.2, 0.2)  # as a fraction for x, y, z axes
    max_shear = (5, 5, 5,5, 5, 5)  # each value for yz, xz, xy planes
    max_shift = (0.15, 0.15, 0.15)  # as a fraction for x, y, z axes
    p = 1.0  # probability to apply transform


    image_transform = BatchImageRandomAffine(
        image_size=image_size,
        spacing=spacing,
        max_rotation=max_rotation,
        max_zoom=max_zoom,
        max_shear=max_shear,
        max_shift=max_shift,
        p=p,
        dtype=torch.float32,
        device=torch.device("cpu"),
        align_corners=align_corners,
    )

    transformed_image_np = image_transform(image).numpy()

    noise = np.random.normal(0, 10.0, transformed_image_np.shape)  
    noisy_image = transformed_image_np + noise

    return np.squeeze(noisy_image)

    #return (torch.from_numpy(transformed_image_np)
    #        .to(dtype=torch.float64))  #shape: (1,1,256,256,50)

def get_prob_logits(logits_list_1, logits_list_2, logits_list_3):
    max_seq_len_1 = max([logits.shape[1] for logits in logits_list_1])
    max_seq_len_2 = max([logits.shape[1] for logits in logits_list_2])
    max_seq_len_3 = max([logits.shape[1] for logits in logits_list_3])

    #max_seq_len = max(max_seq_len_1, max_seq_len_2, max_seq_len_3)

    probs_list_1, probs_list_2, probs_list_3 = [], [], []


    for logits in logits_list_1:
        probs = F.softmax(logits, dim=-1)
        padding_size = max_seq_len_1 - probs.shape[1]
        padded_probs = F.pad(probs, (0, 0, 0, padding_size))  
        probs_list_1.append(padded_probs)


    for logits in logits_list_2:
        probs = F.softmax(logits, dim=-1)
        padding_size = max_seq_len_2 - probs.shape[1]
        padded_probs = F.pad(probs, (0, 0, 0, padding_size)) 
        probs_list_2.append(padded_probs)


    for logits in logits_list_3:
        probs = F.softmax(logits, dim=-1)
        padding_size = max_seq_len_3 - probs.shape[1]
        padded_probs = F.pad(probs, (0, 0, 0, padding_size))  
        probs_list_3.append(padded_probs)

    return probs_list_1, probs_list_2, probs_list_3

def get_token_embeddings(embeddings_list_1, embeddings_list_2, embeddings_list_3):
    max_seq_len = max(
        max([embeddings.shape[1] for embeddings in embeddings_list_1]),
        max([embeddings.shape[1] for embeddings in embeddings_list_2]),
        max([embeddings.shape[1] for embeddings in embeddings_list_3])
    )
    #max_embedding_dim = max(
    #    max([embeddings.shape[2] for embeddings in embeddings_list_1]),
    #    max([embeddings.shape[2] for embeddings in embeddings_list_2]),
    #    max([embeddings.shape[2] for embeddings in embeddings_list_3])
    #)

    def pad_embeddings(embeddings_list):
        padded_list = []
        for embeddings in embeddings_list:
            seq_padding = max_seq_len - embeddings.shape[1]
            #dim_padding = max_embedding_dim - embeddings.shape[2]
            padded_embeddings = F.pad(embeddings, (0, 0, 0, seq_padding))
            padded_list.append(padded_embeddings)
        return padded_list
    embd_1 = pad_embeddings(embeddings_list_1)
    embd_2 = pad_embeddings(embeddings_list_2)
    embd_3 = pad_embeddings(embeddings_list_3)

    return embd_1, embd_2, embd_3

def compute_variance(embd):

    logits_stack = torch.stack(embd, dim=0)  # (num_tta, batch_size, seq_len, vocab_size)
    logits_variance = torch.var(logits_stack, dim=0, unbiased=True)  
    return logits_variance

def compute_covariance(embd_1, embd_2):
    stack_1 = torch.stack(embd_1, dim=0)  # (num_tta, batch_size, seq_len, embedding_dim)
    stack_2 = torch.stack(embd_2, dim=0)  # (num_tta, batch_size, seq_len, embedding_dim)

    num_tta, batch_size, seq_len, embedding_dim = stack_1.shape
    covariance_result = torch.zeros(batch_size, seq_len, embedding_dim, device=stack_1.device)

    for b in range(batch_size):
        for s in range(seq_len):
            embd_1_token = stack_1[:, b, s, :]  
            embd_2_token = stack_2[:, b, s, :] 

            mean_embd_1 = embd_1_token.mean(dim=0)  
            mean_embd_2 = embd_2_token.mean(dim=0)  

            centered_embd_1 = embd_1_token - mean_embd_1  
            centered_embd_2 = embd_2_token - mean_embd_2  

            for e in range(embedding_dim):
                cov = (centered_embd_1[:, e] @ centered_embd_2[:, e]) / (num_tta - 1) 
                covariance_result[b, s, e] = cov 

    return covariance_result.float().detach().cpu().numpy()

def compute_entropy(probs):
    return -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)

def compute_tta_metrics(logits_list_1,logits_list_2,logits_list_3, input_ids
                        ,embde_1,embde_2,embde_3):
    probs_list_1,probs_list_2,probs_list_3 = (
        get_prob_logits(logits_list_1,logits_list_2,logits_list_3))
    embd_1,embd_2,embd_3 = get_token_embeddings(embde_1,embde_2,embde_3)

    metric_df = pd.DataFrame()
    i = 1
    for probs_list,embd in zip([probs_list_1,probs_list_2,probs_list_3],

      probs_mean = torch.mean(torch.stack(probs_list), dim=0)  # (batch_size, seq_len, vocab_size)

      predictive_entropy = compute_entropy(probs_mean)  # (batch_size, seq_len)
      metric_df['Predictive_Etropy_{}'.format(i)] = [predictive_entropy.mean().item()]  

      conditional_entropies = torch.stack([
        compute_entropy(probs) for probs in probs_list  
      ])  # (num_tta, batch_size, seq_len)
      avg_conditional_entropy = conditional_entropies.mean(dim=0)  # (batch_size, seq_len)
      metric_df['Conditional_Etropy_{}'.format(i)] = [avg_conditional_entropy.mean().item()]  

    # === 计算互信息（MI） ===
    #mutual_information = predictive_entropy - avg_conditional_entropy  # (batch_size, seq_len)
    #avg_mutual_information = mutual_information.mean().item()  

      #token_probs_mean = probs_mean.gather(dim=2, index=input_ids.unsqueeze(-1)).squeeze(-1)  # (batch_size, seq_len)
      #token_probs_mean = torch.clamp(token_probs_mean, min=1e-9)  
      #g_nll_pred = -torch.log(token_probs_mean).mean(dim=1).mean().item()  
      #metric_df['Predictive_NLL_{}'.format(i)] = [g_nll_pred]

      #g_nll_conditional = []
      #for probs in probs_list:
        #token_probs = probs.gather(dim=2, index=input_ids.unsqueeze(-1)).squeeze(-1)  # (batch_size, seq_len)
        #token_probs = torch.clamp(token_probs, min=1e-9)  
        #g_nll = -torch.log(token_probs).mean(dim=1)  
        #g_nll_conditional.append(g_nll)
      #g_nll_conditional = torch.stack(g_nll_conditional, dim=0).mean(dim=0).mean().item()  
      #metric_df['Predictive_NLL_{}'.format(i)] = [g_nll_conditional]

      variance = compute_variance(embd)
      metric_df['Variance_{}'.format(i)] = [(variance.float().detach().cpu().numpy())*10e3]

      i += 1

    metric_df['Co_Variance_12'] = [compute_covariance(embd_1, embd_2)*10e3]
    #metric_df['Co_Variance_13'] = [compute_covariance(embd_1, embd_3)]
    #metric_df['Co_Variance_23'] = [compute_covariance(embd_2, embd_3)]
    return metric_df

def process_nii_to_npy(image_path, output_path, Image_aug,
                       affine,affine_size,
                       affine_time,
                       target_depth=32,
                       target_shape=(256, 256)):
    """
    Process a .nii file and convert it to a .npy file with shape 1*target_depth*target_shape[0]*target_shape[1].

    Args:
        image_path (str): Path to the input .nii file.
        output_path (str): Path to save the processed .npy file.
        target_depth (int): Target depth for the 3D image.
        target_shape (tuple): Target height and width for the 3D slices (default is 256x256).

    Returns:
        str: Path to the saved .npy file.
    """
    # Step 1: Load the .nii file
    nii_image = nib.load(image_path)
    image_data = nii_image.get_fdata()  # Extract data as a numpy array
    print(f"Original shape: {image_data.shape}")
    #image_data = image_data[:,:,:]

    # Step 1.5: Aug
    #if Image_aug == True and len(image_data.shape) == 3:
    #    output_path = output_path + '_{}'.format(affine_time)
    #    image_data = image_data[:,:,:]

    if Image_aug == True: #and len(image_data.shape) == 4:
        output_path = output_path + '_{}'.format(affine_time)
        random_sample = random.choice(range(image_data.shape[3]))
        image_data = image_data[:,:,:,random_sample]
        #Affine
        if affine == True:
          image_data = data_affine(image_data)
        print(image_data.shape)
    if Image_aug == False: #and len(image_data.shape) == 4:
        image_data = image_data[:, :, :, 0]

        # Step 1.5: Affine
    #if affine == True:
        #output_path = output_path + '_{}'.format(affine_time)
    #    random_sample = random.sample(range(0, 50), affine_size)
    #    for t in random_sample:
    #        image_data[:, :, t] = data_affine(image_data, t)

    # Step 2: Resize depth
    depth_original = image_data.shape[2]
    resize_factor_depth = target_depth / depth_original
    # Resize height and width to match target_shape
    resize_factors = (target_shape[0] / image_data.shape[0],
                      target_shape[1] / image_data.shape[1],
                      resize_factor_depth)
    image_resized = zoom(image_data, resize_factors, order=1)  # Linear interpolation
    print(f"Shape after resize: {image_resized.shape}")

    # Step 3: Normalize to 0-1
    image_min = image_resized.min()
    image_max = image_resized.max()
    image_normalized = (image_resized - image_min) / (image_max - image_min)
    image_normalized.fill(0)
    print(f"Data range after normalization: {image_normalized.min()} to {image_normalized.max()}")

    # Step 4: Add batch dimension
    image_final = image_normalized[np.newaxis, :, :, :]  # Add batch dimension
    image_final = np.transpose(image_final, (0, 3, 1, 2))  # Rearrange axes to 1×32×256×256
    print(f"Final shape: {image_final.shape}")

    # Step 5: Save as .npy
    np.save(output_path, image_final)
    print(f"Saved processed image to {output_path}")

    return output_path

# Prepare your 3D medical image:
# 1. The image shape needs to be processed as 1*32*256*256, consider resize and other methods.
# 2. The image needs to be normalized to 0-1, consider Min-Max Normalization.
# 3. The image format needs to be converted to .npy
# 4. Although we did not train on 2D images, in theory, the 2D image can be interpolated to the shape of 1*32*256*256 for input.

def get_npy(item_dir_list, name,name_dir_list,Image_aug,affine_time):
    for im_path,name_img in zip(item_dir_list,name_dir_list):
      if name == name_img:
        if Image_aug == False:
         sax_img = name+'_sax.nii.gz'
        if Image_aug == True:
         choices = ['_sax.nii.gz']
         sax_img = name+ random.choice(choices)
        image_path = os.path.join(im_path,sax_img)
        output_path = os.path.join('/SAN/medic/candi_tyc/Yucheng/npy/', 'npy_data')
        output_path = process_nii_to_npy(image_path, output_path, Image_aug,
                                         affine = True,
                                         affine_size = 50, 
                                         affine_time = affine_time,
                                         target_depth=32,
                                         target_shape=(256, 256))
        image_path = output_path
        return image_path


def get_text_info(text_dir,time):
    df = pd.read_excel(text_dir)
    text_name_list = df['eid'].tolist()
    question = df['question'].tolist()
    answer = df['answer'].tolist()
    Cate = df['category'].tolist()
    sampled_data = []
    if time == 1:
      filtered_df = df[~df['answer'].str.contains(r'No|none of above', case=False, na=False)]
      sampled_data = filtered_df.to_dict('records')
      #抽取40%的样本
      sample_size = max(1, int(len(sampled_data) * 0.2)) 
      sampled_data = random.sample(sampled_data, sample_size)
      #answer_counts = Counter(answer)
      #for ans_category in answer_counts.keys():
      #  category_data = df[df['answer'] == ans_category]
      #  sampled_data.extend(category_data.sample(n=min(10, len(category_data)), random_state=42).to_dict('records'))
    else:
        sampled_data = df.to_dict('records')
    return sampled_data

def get_pre(model,tokenizer,proj_out_num,sampled_data,
            item_dir_list,name_dir_list,background_dir,
            save_result_dir):
    i = 0
    for n in sampled_data:
      for time in [1,2,3]:
        if time == 1: 
          Image_aug, Text_aug = True, False
        if time == 2:
          Image_aug, Text_aug = False, True
        if time == 3:
          Image_aug, Text_aug = True, True
        name = n['eid']
        image_path_list = []
        Question = []
        if Image_aug == True:
          image_path_list.append(get_npy(item_dir_list, name,
                                         name_dir_list,False,0))
          for time in np.arange(20): 
              image_path_list.append(get_npy(item_dir_list, name,
                                             name_dir_list,True,
                                             time))
        else: image_path_list.append(get_npy(item_dir_list, name,
                                         name_dir_list,False,0))


        if Text_aug == True:
            Question.append(n['question'])
            Question = sample_text_augmentation(n['question'],Question,
                                    background_dir,eid = name,time=20)
        else:
          Question.append(n['question'])

        image_tokens = "<im_patch>" * proj_out_num


        if Image_aug == True and Text_aug == False:
          time = 0
          logits_list_1 = []
          embd_1 = []
          for image_path in image_path_list:
            image_np = np.load(image_path+'.npy')
            image_pt = torch.from_numpy(image_np).unsqueeze(0).to(dtype=dtype, device=device)
            input_txt =  (f"The sax MRI images are as follows: {image_tokens}, it has 32 time dimensions. "
                        f"{Question[0]} You can select more than 1 correct answers")
            input_id = tokenizer(input_txt, return_tensors="pt")['input_ids'].to(device=device)
            #pad_token_id = tokenizer.pad_token_id
            #attention_mask = (input_id != pad_token_id).long()
            with torch.no_grad():
              generation = model.generate(image_pt, input_id,max_new_tokens=64, do_sample=True, top_p=0.9, temperature=1.0)
              outputs = model(image_pt, input_id) #attention_mask=attention_mask
              #logits_list_1.append(outputs.logits)

            generated_texts = tokenizer.batch_decode(generation, skip_special_tokens=True)
            #generated_input_ids = tokenizer(generated_texts, return_tensors="pt")['input_ids'].to(device=device)
            #generated_embeddings = model.get_input_embeddings()(generated_input_ids)
            #embd_1.append(generated_embeddings)
            print('question', Question[0])
            print('generated_texts', generated_texts)
            n['pre_{}'.format(time)] = generated_texts
            print(time)
            time += 1


        if Text_aug == True and Image_aug == False:
            time = 21
            logits_list_2 = []
            embd_2 = []
            for question in Question:
                image_np = np.load(image_path_list[0] + '.npy')
                image_pt = torch.from_numpy(image_np).unsqueeze(0).to(dtype=dtype, device=device)
                input_txt = (f"The sax MRI images are as follows: {image_tokens}, it has 32 time dimensions. "
                             f"{question} You can select more than 1 correct answers")
                input_id = tokenizer(input_txt, return_tensors="pt")['input_ids'].to(device=device)
                #pad_token_id = tokenizer.pad_token_id
                #attention_mask = (input_id!= pad_token_id).long()
                with torch.no_grad():
                    generation = model.generate(image_pt, input_id, max_new_tokens=64, do_sample=True, top_p=0.9,
                                                temperature=1.0)
                    outputs = model(image_pt, input_id)
                    #logits_list_2.append(outputs.logits)


                generated_texts = tokenizer.batch_decode(generation, skip_special_tokens=True)
                #generated_input_ids = tokenizer(generated_texts, return_tensors="pt")['input_ids'].to(device=device)
                #generated_embeddings = model.get_input_embeddings()(generated_input_ids)
                #embd_2.append(generated_embeddings)
                print('question', question)
                print('generated_texts', generated_texts)
                n['pre_{}'.format(time)] = generated_texts
                print(time)
                time += 1


        if Image_aug == True and Text_aug == True:
            time = 42
            logits_list_3 = []
            embd_3 = []
            for image_path,question in zip(image_path_list,Question):
                image_np = np.load(image_path + '.npy')
                image_pt = torch.from_numpy(image_np).unsqueeze(0).to(dtype=dtype, device=device)
                input_txt = (f"The sax MRI images are as follows: {image_tokens}, it has 32 time dimensions. "
                             f"{question} You can select more than 1 correct answers")
                input_id = tokenizer(input_txt, return_tensors="pt")['input_ids'].to(device=device)
                #pad_token_id = tokenizer.pad_token_id
                #attention_mask = (input_id != pad_token_id).long()
                with torch.no_grad():
                    generation = model.generate(image_pt, input_id, max_new_tokens=64, do_sample=True, top_p=0.9,
                                                temperature=1.0)
                    outputs = model(image_pt, input_id)
                    #logits_list_3.append(outputs.logits)

                generated_texts = tokenizer.batch_decode(generation, skip_special_tokens=True)
                #generated_input_ids = tokenizer(generated_texts, return_tensors="pt")['input_ids'].to(device=device)
                #generated_embeddings = model.get_input_embeddings()(generated_input_ids)
                #embd_3.append(generated_embeddings)
                print('question', question)
                print('generated_texts', generated_texts)
                n['pre_{}'.format(time)] = generated_texts
                print(time)
                time += 1

            #metric_df = compute_tta_metrics(logits_list_1,logits_list_2, logits_list_3,
            #    input_id,embd_1,embd_2,embd_3)
            nn= pd.concat([pd.DataFrame(n)], axis=1) #nn= pd.concat([pd.DataFrame(n), metric_df], axis=1)
            #append_to_json(save_result_dir, nn)
            append_to_excel(save_result_dir, nn)
            #del logits_list_1, logits_list_2, logits_list_3
            #del embd_1, embd_2, embd_3

        # Delete variables to free up memory
        del image_np
        del image_pt
        del generation,outputs
        del generated_texts#,generated_input_ids,generated_embeddings

        # Clear CUDA memory (if using GPU)
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        # Optional: force garbage collection
        gc.collect()

        i += 1
        print(i)
        #if i>1000:
          #break
    return sampled_data

def get_metric(sampled_data):
    #MS = []
    #PREC = []
    #RE = []
    #F1 = []
    #TKA = []
    for i in sampled_data:
      answer = i['answer']
      pre = i['pre']
      ms = exact_match_score(answer,pre)
      prec,re,f1 = precision_recall_f1_score(answer,pre)
      tka = top_k_accuracy(answer,pre)
      i['Ms'] = ms
      i['Prec'] = prec
      i['Re'] = re
      i['F1'] = f1
      i['Tka'] = tka
      #MS.append(ms)
      #PREC.append(prec)
      #RE.append(re)
      #F1.append(f1)
      #TKA.append(tka)

    return sampled_data

#image
#all_data_dir = "/cluster/project7/cardiac_71702_processed/tfds/downloads/manual/processed"
#item_dir_list,name_dir_list = find_data_dir(all_data_dir)
#save_list_dir = r"D:\NewPythonProject\Yucheng\data\img_dir_text.xlsx"
save_list_dir = "/SAN/medic/candi_tyc/Yucheng/data/img_dir_text.xlsx"
#df1 = pd.DataFrame({
#    'Item': item_dir_list,
#    'Name': name_dir_list
#})
#df1.to_excel(save_list_dir, index=False, engine='openpyxl')

print('Image process finished')

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

#model = model.to(device=device)
print('Model loading finished')

#text
background_dir = "/text/background"
#save_result_dir = "/results.json"
save_result_dir = "/results_text_only.xlsx"
df2 = pd.read_excel(save_list_dir)
item_dir_list = df2['Item'].tolist()
name_dir_list = df2['Name'].tolist()
text_dir = '/SAN/medic/candi_tyc/Yucheng/text/evaluation.xlsx'
#text_dir = '/SAN/medic/candi_tyc/Yucheng/text/evaluation_reduced.xlsx'
sampled_data = get_text_info(text_dir,1)

#for time in [1,2,3]: 
#  if time == 1: 
#     Image_aug, Text_aug = True, False
#  if time == 2:
#     Image_aug, Text_aug = False, True
#  if time == 3:
#    Image_aug, Text_aug = True, True
sampled_data = get_pre(model, tokenizer, proj_out_num, sampled_data,
                        item_dir_list, name_dir_list,
                        background_dir, save_result_dir)
    
print(sampled_data.__len__())


#predict and evaluate
#IMG_Path_dir = r"D:\NewPythonProject\Yucheng\text\IMG"
#background_dir = r"D:\NewPythonProject\Yucheng\text\background"
#sampled_data = get_metric(sampled_data)

#df3 = pd.DataFrame(sampled_data)
#df3.to_excel(save_result_dir, index=False, engine='openpyxl')
#print('Results Saved')
