# reconstructing medium 3D face
CUDA_VISIBLE_DEVICES=1 python -u main.py \
    --mode=test \
    --data_dir=./results/de_occlusion \
    --batch_size=1 \
    --bfm_path="resources/BFM2009_Model.mat" \
    --ver_uv_index="resources/vertex_uv_ind.npz" \
    --uv_face_mask_path="resources/face_mask.png" \
    --vgg_path="resources/vgg-face.mat" \
    --load_coarse_ckpt=./checkpoints/Reconstruction/coarse-model/coarse-resnet \
    --load_medium_ckpt=./checkpoints/Reconstruction/medium-model/medium \
    --is_medium_model \
    --output_dir=./results/reconstruction
