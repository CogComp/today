
export Seed=18
export Step=1000
export DATASET='combine' #matres #tracie
export NEW_MODEL_NAME="model_wo_today"


export MODEL_NAME="../model/ptntime-pretrained-model"
export OUTPUT_DIR="../model/${NEW_MODEL_NAME}"

# Hard-Label Training set
# If DATASET == 'combine'
export TRAIN_FILE="../data/combine/combined_train.txt"
# If DATASET == 'matres'
# export TRAIN_FILE="../data/matres/matres_train.txt"
# If DATASET == 'tracie'
# export TRAIN_FILE="../data/tracie/tracie_train.txt"




# Training
python ./train_t5.py \
   --output_dir=$OUTPUT_DIR \
   --model_type=t5 \
   --tokenizer_name=t5-large \
   --model_name_or_path=$MODEL_NAME \
   --do_train \
   --num_train_epochs=5 \
   --max_steps=$Step\
   --train_data_file=$TRAIN_FILE \
   --line_by_line \
   --per_gpu_train_batch_size=1 \
   --per_device_train_batch_size=1 \
   --gradient_accumulation_steps=16 \
   --per_device_eval_batch_size=1 \
   --per_gpu_eval_batch_size=1 \
   --save_steps=5000 \
   --logging_steps=1000 \
   --overwrite_output_dir \
   --seed=$Seed
