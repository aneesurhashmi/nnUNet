# nnUNetv2_predict -d 001 3d_fullres -f  -tr nnUNetTrainer_250epochs
fold="0"
inference_dir="/dhc/home/anees.hashmi/bowel-segmentation/nnUNet/inference_output/Dataset001_BowelSeg"

# nnUNetv2_predict -d 001 -tr nnUNetTrainer_250epochs -c 3d_fullres -i /dhc/home/anees.hashmi/data/nnUNet_raw/Dataset001_BowelSeg/imagesTs -o inference_output/imagesTs_prednnUNET250 -f ${fold}

# nnUNetv2_evaluate_folder /dhc/home/anees.hashmi/data/nnUNet_raw/Dataset001_BowelSeg/labelsTs ${inference_dir}/nnUNET250/fold_0 -djfile ${inference_dir}/nnUNET250/fold_${fold}/dataset.json -pfile ${inference_dir}nnUNET//fold_${fold}/plans.json

# for fold in {1..4}; do
    # echo "PREDICTING FOLD: ${fold}"
    # nnUNetv2_predict -d 001 -tr nnUNetTrainer_250epochs -c 3d_fullres -i /dhc/home/anees.hashmi/data/nnUNet_raw/Dataset001_BowelSeg/imagesTs -o ${inference_dir}/nnUNET250/fold_${fold} -f ${fold}
# done

for fold in {1..4}; do
    echo "EVALUATING FOLD: ${fold}"
    nnUNetv2_evaluate_folder /dhc/home/anees.hashmi/data/nnUNet_raw/Dataset001_BowelSeg/labelsTs ${inference_dir}/nnUNET250/fold_${fold} -djfile ${inference_dir}/nnUNET250/fold_${fold}/dataset.json -pfile ${inference_dir}/nnUNET250/fold_${fold}/plans.json
done