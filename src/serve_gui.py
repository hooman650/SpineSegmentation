"""

This is a simple streamlit app to serve the trained neural network segmentation for testing.  

Example:
    In order to invoke this module::
        >>> streamlit run serve_gui.py


Dependencies:
    * model
   
Log:
    * Initial release (Hooman Sedghamiz, Jan, 2022)
    
# Copyright 2019-present, the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
    
    
"""

# --------- UI Public dependencies ------------#
import streamlit as st
import numpy as np
import os
import matplotlib.pyplot as plt
import torch as t
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import argparse
from utils.utils import soft_to_hard_pred, keep_largest_connected_components
from utils.loss import DiceCoefMultilabelLoss
from model.dilated_unet import Segmentation_model
from utils.metric import dice_coef_multilabel
from dataset import ImageProcessor, DataGenerator
import pandas as pd
import torch
import cv2

@st.cache(allow_output_mutation=True, show_spinner=True)
def load_model():
    comments = "Verterbra_disk.unet_lr_0.0001_32.gaussian_noise"
    unet_model = Segmentation_model(filters=32,
                                in_channels=3,
                                n_block=4,
                                bottleneck_depth=4,
                                n_class=3)
    unet_model.load_state_dict(torch.load('./weights/{}/unet_model_checkpoint.pt'.format(comments)))
    
    return unet_model

@st.cache(allow_output_mutation=True, show_spinner=True)    
def predict(unet_model,ids_valid):
    validA_generator = DataGenerator(df=pd.Series([ids_valid]),
                                 channel="channel_first",
                                 apply_noise=False,
                                 phase="valid",
                                 apply_online_aug=False,
                                 batch_size=1,
                                 n_samples=-1)
    id_n = 0
    for dataset in validA_generator:
        x_batch, y_batch = dataset
        print(x_batch.shape, y_batch.shape)
        prediction = unet_model.forward(t.tensor(x_batch).cuda(), features_out=False)
        y_pred = soft_to_hard_pred(prediction.cpu().detach().numpy(), 1)

        print("The validation dice score:", dice_coef_multilabel(y_true=y_batch, y_pred=y_pred, channel='channel_first'))

        y_pred = np.moveaxis(y_pred, 1, -1)
        y_pred = np.argmax(y_pred, axis=-1)
        #y_pred = keep_largest_connected_components(mask=y_pred)

        y_batch = np.moveaxis(y_batch, 1, -1)
        y_batch = np.argmax(y_batch, axis=-1)

        plt.figure()
        plt.imshow(np.moveaxis(x_batch[id_n], source=0, destination=2), cmap='gray')
        plt.axis('off')
        plt.savefig('./results/run_raw.png',bbox_inches='tight',transparent=True)
        plt.figure()
        plt.imshow(y_batch[id_n], cmap='jet')
        plt.axis('off')
        plt.savefig('./results/run_mask.png',bbox_inches='tight',transparent=True)
        plt.figure()
        plt.imshow(y_pred[id_n], cmap='gray')
        plt.axis('off')
        plt.savefig('./results/run_pred.png',bbox_inches='tight',transparent=True)
    
    
def display_results():
    cols = st.columns(3)
    cols[0].info('Input')
    cols[1].success('Ground Truth')
    cols[2].success('Prediction')
    cols = st.columns(3)
    cols[0].image('./results/run_raw.png')
    cols[1].image('./results/run_mask.png')
    cols[2].image('./results/run_pred.png')
    
        


def main_module() -> None:
    """Main dashboard module and constructor.

    """
    # ----------Load menues ----------------#
    st.warning("DISCLAIMER: THIS IS AN EXPERIMENTAL TOOL AND SHOULD NOT BE USED FOR OTHER PURPOSES.") 
    model = load_model()
    file_n = st.selectbox('Select an image',[15,16,17,18,19])
    predict(model,file_n)
    display_results()


if __name__ == "__main__":
    # ------------------ General Stuff --------------------#
    st.set_page_config(
        page_title="SpineAssistant",
        page_icon=":stethoscope:",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.sidebar.image("./imgs/logo.PNG")
    main_module()

    # ------------------ Set Side Bar -------------------- #
    st.sidebar.title("About")
    st.sidebar.info("## SpineAssistant ## \n"
                        "**SpineAssistant** is a ML tool for spine surgery assistant. \n\n"                        
                        " This page is under active development.\n\n If you have any comments forward to: \n\n"
                        "##### mailto:hooman.sedghamiz@gmail.com \n"
                        "##### mailto:snemati@health.ucsd.edu \n"
                        "##### mailto:josorio@health.ucsd.edu")
    