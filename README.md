# Hi3DFace
Tensorflow implementation for the paper Hi3DFace: High-realistic 3D Face Reconstruction from a Single Occluded Image

<p>
<img src="figures/framework.png" alt="framework" width="875px">
</p>

## Requirements
- Linux + Anaconda
- CUDA 10.0 + CUDNN 7.6.0
- Python 3.7
- Tensorflow 1.15.0
- [tf_mesh_renderer](https://github.com/google/tf_mesh_renderer)

## Installation
### ● Clone the repository
```bash
git clone https://github.com/sys706/Hi3DFace.git
cd Hi3DFace
```

### ● Set up the environment
If you use anaconda, run the following:
```bash
conda create -n hi3dface python=3.7
conda activate hi3dface
pip install -r requirements.txt
```

### ● Compile tf_mesh_renderer
```bash
TF_INC=./env/lib/python3.7/site-packages/tensorflow_core/include
TF_LIB=./env/lib/python3.7/site-packages/tensorflow_core
```
you might need the following to successfully compile the third-party library
```bash
ln -s ./env/lib/python3.7/site-packages/tensorflow_core/libtensorflow_framework.so.1 ./env/lib/python3.7/site-packages/tensorflow_core/libtensorflow_framework.so
```

compiles the c++ kernel of the differentiable renderer.
```bash
mkdir ./tools/kernels
g++ -std=c++11 \
    -shared ./tools/src_mesh_renderer/rasterize_triangles_grad.cc ./tools/src_mesh_renderer/rasterize_triangles_op.cc ./tools/src_mesh_renderer/rasterize_triangles_impl.cc ./tools/src_mesh_renderer/rasterize_triangles_impl.h \
    -o ./tools/kernels/rasterize_triangles_kernel.so -fPIC -D_GLIBCXX_USE_CXX11_ABI=0 \
    -I$TF_INC -L$TF_LIB -ltensorflow_framework -O
```
Note that find the correct path to **TF_INC** and **TF_LIB**. If it does not work, please try to find them manually. You can also compile the codes using the approach provided by [**tf_mesh_renderer**]().

## Run de-occlusion model
- Download **[vgg16.npy](https://drive.google.com/file/d/1aJuYcsRbz3XssHpIa8zBTHbru8IvsxVH/view?usp=drive_link)** and put it under ```DeOcclusion/vgg```.
- Download the pre-trained models: **[eyeglass](https://drive.google.com/file/d/1w7pz5FHZN8_G5QJxkzN96wn-55P476cY/view?usp=sharing)** | **[hand](https://drive.google.com/file/d/1eiEyzsNkv-TsSFBCUTpG27gbe-hJeuvl/view?usp=sharing)** | **[hat](https://drive.google.com/file/d/13Vfk15yHsnRdQav6kyvroka3sEhkV0Xx/view?usp=sharing)** | **[microphone](https://drive.google.com/file/d/1HPKtB2X4R7xiSc3Z9y759-wjYb1XKQpX/view?usp=sharing)**, unzip and put them under the directory ```checkpoints/DeOcclusion```.
- Test the eyeglass model, run the following:
```bash
python test.py --output '../results/de_occlusion' --test_data_path '../inputs/eyeglass.png' --mask_path '../inputs/eyeglass_mask.png' --model_path '../checkpoints/DeOcclusion/eyeglass/eyeglass'
```
- Test the hand model, run the following:
```bash
python test.py --output '../results/de_occlusion' --test_data_path '../inputs/hand.png' --mask_path '../inputs/hand_mask.png' --model_path '../checkpoints/DeOcclusion/hand/hand'
```
- Test the hat model, run the following:
```bash
python test.py --output '../results/de_occlusion' --test_data_path '../inputs/hat.png' --mask_path '../inputs/hat_mask.png' --model_path '../checkpoints/DeOcclusion/hat/hat'
```
- Test the microphone model, run the following:
```bash
python test.py --output '../results/de_occlusion' --test_data_path '../inputs/micro.png' --mask_path '../inputs/micro_mask.png' --model_path '../checkpoints/DeOcclusion/micro/micro'
```

## Run face reconstruction model from the above de-occluded images

Reconstructing smooth, medium-scale and fine-scale faces from the above de-occluded images,  respectively.
- Download **[resources.zip](https://drive.google.com/file/d/1gkG6rw9zu9vxfkuUo26Pczc2v2SyoEvH/view?usp=sharing)**, unzip and put it under the directory ```resources/```.
- Download the pre-trained coarse model **[coarse-model.zip]()**, medium model **[medium-model.zip]()**, and fine model **[fine-model.zip]()**, unzip and put them under the directory ```checkpoints/Reconstruction```.
- Reconstruct smooth faces, run the following:
```bash
./run_test_coarse.sh
```
- Reconstruct medium-scale faces with detailed wrinkles, run the following:
```bash
./run_test_medium.sh
```
- Reconstruct fine-scale faces with high-fidelity textures that are close to the input images, run the following:
```bash
./run_test_fine.sh
```



