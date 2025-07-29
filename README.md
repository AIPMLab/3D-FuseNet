# 3D-FuseNet

## File Name

### Training

- `main_sf.py`: Traning 3D-FuseNet backbone
- `main_unet.py`: Training U-Net backbone
- `main_res.py`: Training Resnet50 backbone
- `main_unet.py`: Training U-Net backbone

- `main_sf_kf.py`: 5-fold cross-validation

### Result

- `get_out_clinical copy.py`: Clinical result
- `get_out_system copy.py`: Result of model using deep features with different optimizer and ML model
- `get_out_system copy 3.py`: Result of 3D-FuseNet with different optimizer and ML model
- `get_out_system copy 4.py`: Result of other backbone with different ML model
- `get_out copy 5.py`: Result of deep model with different learning

### Model Structure

- `segformer3d.py`: Main structure of 3D-FuseNet
- `image_encoder3d.py`: VisionTransformer backbone
- `resnet.py`: Resnet-50 backbone
- `unet3d.py`: U-Net backbone
