import streamlit as st
import pandas as pd
from PIL import Image
import subprocess
import os
import base64
import pickle
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.preprocessing import StandardScaler
import random
from tqdm import tqdm
import esm
import time


def ESM_feature(sequence):
    esm_model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()  # 320d
    batch_converter = alphabet.get_batch_converter()
    esm_model.eval()  # disables dropout for deterministic results
    data = []
    for i in tqdm(sequence):
        row = (i, i)
        data.append(row)
    # Load ESM-2 model
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    with torch.no_grad():
        results = esm_model(batch_tokens, repr_layers=[6], return_contacts=True)
    token_representations = results["representations"][6]
    # Generate per-sequence representations via averaging
    # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
    sequence_representations = []
    for i, tokens_len in enumerate(batch_lens):
        sequence_representations.append(
            token_representations[i, 1 : tokens_len - 1].mean(0)
        )
    return sequence_representations


class TransformerModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TransformerModel, self).__init__()
        # 构建Transformer编码层，参数包括输入维度、注意力头数
        # 其中d_model要和模型输入维度相同
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_size, batch_first=True, nhead=8  # 输入维度
        )  # 注意力头数
        # 构建Transformer编码器，参数包括编码层和层数
        self.encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=10  # 编码层
        )  # 层数
        # 构建线性层，参数包括输入维度和输出维度（num_classes）
        self.fc = nn.Linear(input_size, num_classes)  # 输入维度  # 输出维度

    def forward(self, x):
        # print("A:", x.shape)  # torch.Size([142, 13])
        x = x.unsqueeze(1)  # 增加一个维度，变成(batch_size, 1, input_size)的形状
        # print("B:", x.shape)  # torch.Size([142, 1, 13])
        x = self.encoder(x)  # 输入Transformer编码器进行编码
        # print("C:", x.shape)  # torch.Size([142, 1, 13])
        x = x.squeeze(1)  # 压缩第1维，变成(batch_size, input_size)的形状
        # print("D:", x.shape)  # torch.Size([142, 13])
        x = self.fc(x)  # 输入线性层进行分类预测
        # print("E:", x.shape)  # torch.Size([142, 3])
        return x


# File download
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="prediction.csv">Download Predictions</a>'
    return href


# Logo image
image = Image.open('cover.png')

st.image(image, use_column_width=True)


# Page title
st.markdown(
    """
# ACE-I inhibitory peptide Predictor (AHTPeptideFusion)

This app allows you to predict ACE-I inhibitory peptide.

**Credits**
- AHTPeptideFusion：A segmented fusion ACE-I inhibitory peptide predictor based on protein language model and deep learning.
- Peptide descriptor calculated using ESM-2.
- AHTPeptideFusion will be a powerful ACE-I inhibitor peptide prediction tool,it can help scientists to accelerate the mining and design of ACE-I inhibitory peptide and reduce the cost of experimental and research and development (R&D) cycle.
"""
)

# Sidebar

with st.sidebar.header('1. Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader("OR, upload your input file", type=['csv'])
    st.sidebar.markdown(
        """
[Example input file](https://github.com/panernie/AHTPeptideFusion/blob/main/test192.csv)
"""
    )

    title = st.text_input("Input your sequence, eg. IRW")

if st.sidebar.button('Predict'):
    T1 = time.time()
    seed_value = 42
    random.seed(seed_value)  # 设置 random 模块的随机种子
    np.random.seed(seed_value)  # 设置 numpy 模块的随机种子
    torch.manual_seed(seed_value)  # 设置 PyTorch 中 CPU 的随机种子
    # tf.random.set_seed(seed_value)                 # 设置 Tensorflow 中随机种子
    if torch.cuda.is_available():  # 如果可以使用 CUDA，设置随机种子
        torch.cuda.manual_seed(seed_value)  # 设置 PyTorch 中 GPU 的随机种子
    # 检测GPU是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scaler_filename = r"model\Standard_scaler.save"
    RF_model = joblib.load(r"model\RandomForest_classf.pkl")
    model = torch.load(
        r'model\transformer_class.pt', map_location=torch.device('cuda:0')
    )  # .to(device)
    if len(title) == 0:
        df = pd.read_csv(uploaded_file)

        st.header('**Original input data**')
        st.write(f"{df.shape[0]}peptide were identified")

        with st.spinner("Please wait..."):
            time.sleep(1)
            # load_data.to_csv('molecule.smi', sep = '\t', header = False, index = False)
            len_num = [len(num) for num in df["sequence"]]
            df["len_num"] = len_num
            try:
                RF_sequence = df[df["len_num"] > 10]["sequence"]
                transformer_sequence = df[df["len_num"] < 11]["sequence"]
                RF_sequence_representations = ESM_feature(RF_sequence)
                transformer_sequence_representations = ESM_feature(transformer_sequence)
                scaler = joblib.load(scaler_filename)
                RF_pred = torch.stack(RF_sequence_representations)
                transformer_X_pred = torch.stack(transformer_sequence_representations)
                RF_X_pred = scaler.transform(RF_pred)
                transformer_X_pred = scaler.transform(transformer_X_pred)
                RF_y = RF_model.predict(RF_pred)
                RF_y_pro = RF_model.predict_proba(RF_pred)
                RF_df = pd.DataFrame(np.array(RF_sequence), columns=["sequence"])
                RF_df["Active_pro"] = RF_y_pro[:, 0]
                RF_df["Inactive_pro"] = RF_y_pro[:, 1]
                with torch.no_grad():
                    model.eval()
                    y_hat = model(
                        torch.tensor(transformer_X_pred).float().to(device)
                    )  # 使用训练好的模型对测试集进行预测
                    y_score = torch.softmax(y_hat, dim=1).data.cpu().numpy()
                    prediction = torch.max(F.softmax(y_hat, dim=1), 1)[1]
                # pred_y = prediction.data.cpu().numpy().squeeze()
                transformer_df = pd.DataFrame(
                    np.array(transformer_sequence), columns=["sequence"]
                )
                transformer_df["Active_pro"] = y_score[:, 0]
                transformer_df["Inactive_pro"] = y_score[:, 1]
                df_all = pd.concat([RF_df, transformer_df], axis=0)
                types = []
                for i in df_all["Active_pro"]:
                    if i < 0.5:
                        types.append("Non-ACE-I inhibitory peptide")
                    else:
                        types.append("ACE-I inhibitory peptide")
                df_all["Type"] = types

                # st.markdown(filedownload(df_all), unsafe_allow_html=True)
            except:
                transformer_sequence = df[df["len_num"] < 11]["sequence"]
                transformer_sequence_representations = ESM_feature(transformer_sequence)
                scaler = joblib.load(scaler_filename)
                transformer_X_pred = torch.stack(transformer_sequence_representations)
                transformer_X_pred = scaler.transform(transformer_X_pred)
                with torch.no_grad():
                    model.eval()
                    y_hat = model(
                        torch.tensor(transformer_X_pred).float().to(device)
                    )  # 使用训练好的模型对测试集进行预测
                    y_score = torch.softmax(y_hat, dim=1).data.cpu().numpy()
                    prediction = torch.max(F.softmax(y_hat, dim=1), 1)[1]
                df_all = pd.DataFrame(
                    np.array(transformer_sequence), columns=["sequence"]
                )
                df_all["Active_pro"] = y_score[:, 0]
                df_all["Inactive_pro"] = y_score[:, 1]
                types = []
                for i in df_all["Active_pro"]:
                    if i < 0.5:
                        types.append("Non-ACE-I inhibitory peptide")
                    else:
                        types.append("ACE-I inhibitory peptide")
                df_all["Type"] = types
    else:
        st.header('**Original input data**')
        st.write(f"1 peptide were identified:", title)

        with st.spinner("Please wait..."):
            time.sleep(1)
            sequence_sig = []
            sequence_sig.append(title)
            df_all = pd.DataFrame(np.array(sequence_sig), columns=["sequence"])
            # load_data.to_csv('molecule.smi', sep = '\t', header = False, index = False)
            transformer_sequence = df_all["sequence"]
            transformer_sequence_representations = ESM_feature(transformer_sequence)
            scaler = joblib.load(scaler_filename)
            transformer_X_pred = torch.stack(transformer_sequence_representations)
            transformer_X_pred = scaler.transform(transformer_X_pred)
            with torch.no_grad():
                model.eval()
                y_hat = model(
                    torch.tensor(transformer_X_pred).float().to(device)
                )  # 使用训练好的模型对测试集进行预测
                y_score = torch.softmax(y_hat, dim=1).data.cpu().numpy()
                prediction = torch.max(F.softmax(y_hat, dim=1), 1)[1]
            df_all["Active_pro"] = y_score[:, 0]
            df_all["Inactive_pro"] = y_score[:, 1]
            if y_score[:, 0] < 0.5:
                df_all["Type"] = "Non-ACE-I inhibitory peptide"
            else:
                df_all["Type"] = "ACE-I inhibitory peptide"

            # st.markdown(filedownload(df_all), unsafe_allow_html=True)
    # df_10 = df_all.iloc[:10]
    file_names = time.time()
    df_all.to_csv(f"log\{file_names}.csv", index=None)
    # print(df_all)
    df_10 = df_all[:10]
    T2 = time.time()
    st.success("Done!")
    st.write('Program run time:%sms' % ((T2 - T1) * 1000))
    st.header('**Output data**')
    st.write("Only the first 10 results are displayed!")
    st.write(df_10)
    st.markdown(filedownload(df_all), unsafe_allow_html=True)
else:
    st.info('Upload input data in the sidebar to start!')
