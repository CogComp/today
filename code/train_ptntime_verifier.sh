
export Seed=18
export Step=1000
export DATASET='combine' #matres #tracie


# Verifier Models
export MODEL_NAME="../model/ptntime-pretrained-model"
export GENERAL_MODEL_NAME="ptntime_general_verifier"
export AS_MODEL_NAME="ptntime_additional_verifier"
export GENERAL_MODEL_DIR="../model/${GENERAL_MODEL_NAME}"
export AS_MODEL_DIR="../model/${AS_MODEL_NAME}"

#Training Data for the Verification System
# General Explanation Verifier
export TRAIN_FILE_GENERAL="../data/today/today_train_with_explanation.txt"
# Additional Sentence Verifier
export TRAIN_FILE_ADD="../data/today/today_train_without_explanation.txt"

# Hard-Label Training set
#If DATASET == 'combine'
export TRAIN_FILE_HARD="../data/combine/combined_train.txt"
#If DATASET == 'matres'
#export TRAIN_FILE_HARD="../data/matres/matres_train.txt"
#If DATASET == 'tracie'
#export TRAIN_FILE_HARD="../data/tracie/tracie_train.txt"




# Train General Explanation Verifier
python ./train_t5_with_today.py \
  --output_dir=$GENERAL_MODEL_DIR \
  --model_type=t5 \
  --tokenizer_name=t5-large \
  --model_name_or_path=$MODEL_NAME \
  --do_train \
  --num_train_epochs=50\
  --max_steps=$Step\
  --train_data_file_1=$TRAIN_FILE_GENERAL \
  --train_data_file_2=$TRAIN_FILE_HARD \
  --line_by_line \
  --per_gpu_train_batch_size=1 \
  --per_device_train_batch_size=1 \
  --gradient_accumulation_steps=16 \
  --per_device_eval_batch_size=1 \
  --per_gpu_eval_batch_size=1 \
  --loss_alpha=10 \
  --loss_beta=1 \
  --save_steps=5000 \
  --logging_steps=1000 \
  --overwrite_output_dir \
  --seed=$Seed

# Train Additional Sentence Verifier
python ./train_t5_with_today.py \
  --output_dir=$AS_MODEL_DIR \
  --model_type=t5 \
  --tokenizer_name=t5-large \
  --model_name_or_path=$MODEL_NAME \
  --do_train \
  --num_train_epochs=50\
  --max_steps=$Step\
  --train_data_file_1=$TRAIN_FILE_ADD \
  --train_data_file_2=$TRAIN_FILE_HARD \
  --line_by_line \
  --per_gpu_train_batch_size=1 \
  --per_device_train_batch_size=1 \
  --gradient_accumulation_steps=16 \
  --per_device_eval_batch_size=1 \
  --per_gpu_eval_batch_size=1 \
  --loss_alpha=10 \
  --loss_beta=1 \
  --save_steps=5000 \
  --logging_steps=1000 \
  --overwrite_output_dir \
  --seed=$Seed


