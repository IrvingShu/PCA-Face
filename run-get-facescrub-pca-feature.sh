export PYTHONPATH=/usr/local/lib/python2.7/dist-packages:$PYTHON


nohup python -u ./src/get_pca_feature.py \
    --image-list=../data/facescrub/facescrub-mtcnn-aligned-112x112-image-list.txt \
    --feature-dir=/workspace/data/facescrub-features/insightface-r100-spa-m2.0-ep96 \
    --feature-dims=xxx
    --ipca-model-path=../model/256_ipca_split0.pkl \
    --save-format=_feat.bin \
    --out-dir=/workspace/data/facescrub-features/pca-insightface-r100-spa-m2.0-ep96/pca-256-split0 \
    > ./logs/pca-256-split0-facescrub-r100-spa-m2.0-ep96.txt &
