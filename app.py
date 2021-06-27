import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
import numpy as np
import torch
from torchvision import transforms as T
from torch import optim
st.set_page_config(
    page_title='Neural Style Transfer'
)
def br(i):
    st.markdown("<br>" * i,unsafe_allow_html=True)

components.html("""
            <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
            <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js"></script>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.16.0/umd/popper.min.js"></script>
            <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>
            """, height=50)

#Function used to load images from both given folder as well as uploaded images

def load_img(im1,im2):
    st.sidebar.markdown("**Upload Images**", unsafe_allow_html=True)
    cont_img = st.sidebar.file_uploader('Upload Content Image',type=["png", "jpg"], key=1)
    sty_img = st.sidebar.file_uploader('Upload Style Image',type=["png", "jpg"], key=2)
    st.sidebar.markdown("***Custom Examples(Only works if no image is uploaded )***", unsafe_allow_html=True)
    content_img = ['any','Steve Jobs', 'Hornbill', 'spiderman']
    style_img = ['any','style10', 'style11', 'style12']
    content_bar = st.sidebar.selectbox(label='Content Image',options= content_img, key=3)
    style_bar = st.sidebar.selectbox(label='Style Image',options= style_img, key=4)
    if content_bar != 'any' and (cont_img is None and sty_img is None):
        im1 = Image.open('images/'+content_bar+'.jpg').convert('RGB')     
    elif cont_img is not None:
        im1 = Image.open(cont_img).convert('RGB')
    if style_bar != 'any' and (cont_img is None and sty_img is None):
        im2 = Image.open('images/'+style_bar+'.jpg').convert('RGB')
    elif sty_img is not None:
        im2 = Image.open(sty_img).convert('RGB')
    
    return im1,im2
st.title('Neural Style Transfer')
br(1)
#Loading the pretrained vgg19 model

model = torch.load('model')
for parameters in model.parameters():
  parameters.requires_grad_(False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == 'cpu':
  st.markdown("<div style='background-color:#780000; color: white; border-radius: 10px; text-indent: 35px; width: 70%'><b><i>Note:</i></b><p>You are using <b>CPU</b> which will consume a lot of time</p>",unsafe_allow_html=True)
model.to(device)

# Preprocess the image

def preprocess(img, max_size = 500):
  image = img

  if max(image.size) > max_size:
    size = max_size
  else:
    size = max(image.size)

  img_transforms = T.Compose([
                              T.Resize(size),
                              T.ToTensor(),
                              T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
  ])

  image = img_transforms(image)

  image = image.unsqueeze(0)
  return image

# Deprocess the image

def deprocess(tensor):

  image = tensor.to('cpu').clone()
  image = image.numpy()
  image = image.squeeze(0)
  image = image.transpose(1,2,0)  
  image = image *  np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
  image = image.clip(0,1)

  return image

# Get the features of image

def get_features(image,model):

  layers = {
      '0' : 'conv1_1',
      '5' : 'conv2_1',
      '10' : 'conv3_1',
      '19' : 'conv4_1',
      '21' : 'conv4_2', #content_feature
      '28' : 'conv5_1'
  }

  x = image
  Features = {}

  for name, layer in model._modules.items():
    x = layer(x)
    if name in layers:
      
      Features[layers[name]] = x
  
  return Features

# Create Gram Matrix

def gram_matrix(tensor):
  b,c,h,w = tensor.size()
  tensor = tensor.view(c, h*w)
  gram = torch.mm(tensor, tensor.t())
  return gram

# Defining Content Loss and Style Loss Functions

def content_loss(target_conv4_2, content_conv4_2):

  loss = torch.mean((target_conv4_2 - content_conv4_2)**2)
  return loss

def style_loss(style_weights, target_features, style_grams):

  loss = 0

  for layer in style_weights:
    target_f = target_features[layer]
    target_gram = gram_matrix(target_f)
    style_gram = style_grams[layer]
    b,c,h,w = target_f.shape
    layer_loss = style_weights[layer] * torch.mean((target_gram - style_gram)**2)
    loss += layer_loss/(c*h*w)
  
  return loss

# Total Loss

def total_loss(c_loss, s_loss, alpha,beta):
  loss = alpha * c_loss + beta * s_loss
  return loss

br(2)
k1,k2 = st.beta_columns(2)
br(2)
im1 = None
im2 = None
cont_fin, sty_fin = load_img(im1,im2)
try:
    k1.image(cont_fin, caption='Content Image', width=200)
except:
    k1.write("No Image")
try:
    k2.image(sty_fin, caption='Style Image', width=200)
except:
    k2.write("No Image")

if st.button('Generate Styled Image'):
    if cont_fin is not None and sty_fin is not None:
        content_p = preprocess(cont_fin)
        style_p = preprocess(sty_fin)
        content_p = content_p.to(device)
        style_p = style_p.to(device)
        content_f = get_features(content_p,model)
        style_f = get_features(style_p,model)
        style_grams = {layer : gram_matrix(style_f[layer]) for layer in style_f}
        style_weights = {
        'conv1_1' : 1.0,
        'conv2_1' : 0.75,
        'conv3_1' : 0.2,
        'conv4_1' : 0.2,
        'conv5_1' : 0.2
        }
        target = content_p.clone().requires_grad_(True).to(device)
        target_f = get_features(target, model)
        optimizer = optim.Adam([target], lr = 0.003)
        alpha = 1
        beta = 1
        epochs = 3000
        show_every = 500
        count = 0
        results = []
        st.write("Loading...")
        for i in range(epochs):
            target_f = get_features(target, model)
            c_loss = content_loss(target_f['conv4_2'], content_f['conv4_2'])
            s_loss = style_loss(style_weights, target_f, style_grams)
            t_loss = total_loss(c_loss, s_loss, alpha, beta)

            optimizer.zero_grad()
            t_loss.backward()
            optimizer.step()

            if i%show_every == 0:
                #st.write("Total Loss at Epoch {}: {}".format(i,t_loss))
                results.append(deprocess(target.detach()))
            if i%30 == 0:
              st.write(f"{count}%")
              count+=1
        target_copy = deprocess(target.detach())
        content_copy = deprocess(content_p)
        k3,k4 = st.beta_columns([0.5,1])
        k4.image(target_copy)
    else:
        st.write('Please check the image formats')

