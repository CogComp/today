
export Seed=18
export Step=1000
export DATASET='combine' #matres #tracie


# Verifier Models
export MODEL_NAME="../model/ptntime-pretrained-model"
export GENERAL_MODEL_NAME="ptntime_general_verifier"
export AS_MODEL_NAME="ptntime_additional_verifier"
export EXP_MODEL_NAME="ptntime_explanation_verifier"
export GENERAL_MODEL_DIR="../model/${GENERAL_MODEL_NAME}"
export AS_MODEL_DIR="../model/${AS_MODEL_NAME}"
export EXP_MODEL_DIR="../model/ptntime_explanation_verifier" #pretrained

# Model Trained with Distilled Data
export NEW_MODEL_NAME="best_ptntime_distill"
export MODEL_OUTPUT_DIR="../model/${NEW_MODEL_NAME}"

# Hard-Label Training set
#If DATASET == 'combine'
export TRAIN_FILE_HARD="../data/combine/combined_train.txt"
#If DATASET == 'matres'
#export TRAIN_FILE_HARD="../data/matres/matres_train.txt"
#If DATASET == 'tracie'
#export TRAIN_FILE_HARD="../data/tracie/tracie_train.txt"

# GPT-3.5 data
export OUTPUT_DIR="../experiment_result/ptntime_distill"
export DATA_FOLDER="../data/gpt3.5/iter0"
export ORIGIN_FILE="gpt3.5_train_roc_implict_with_explanation"
export TEST_FILE_NAME="gpt3.5_train_roc_implict_with_explanation"
export WO_TEST_FILE_NAME="gpt3.5_train_roc_implict_without_explanation"
export EXP_VERIFIER_NAME="gpt3.5_train_roc_implict_explanation_verifier"

export EXP_TEST_FILE="${DATA_FOLDER}/${EXP_VERIFIER_NAME}.txt"
export EXP_SAVE_NAME="${EXP_MODEL_NAME}_${EXP_VERIFIER_NAME}"
export GENERAL_TEST_FILE="${DATA_FOLDER}/${TEST_FILE_NAME}.txt"
export AS_TEST_FILE="${DATA_FOLDER}/${WO_TEST_FILE_NAME}.txt"
export GENERAL_SAVE_NAME="${GENERAL_MODEL_NAME}_${TEST_FILE_NAME}"
export AS_SAVE_NAME="${AS_MODEL_NAME}_${WO_TEST_FILE_NAME}"

# Distill Data
export EXTRA="annotated_train_with_distilled_${GENERAL_MODEL_NAME}_${DATASET}.txt"
export TRAIN_FILE_DISTILL="${OUTPUT_DIR}/${EXTRA}"


# Distill GPT3.5 data
python ./train_t5.py \
    --output_dir=$OUTPUT_DIR \
    --model_type=t5 \
    --tokenizer_name=t5-large \
    --model_name_or_path=$EXP_MODEL_DIR\
    --do_eval \
    --eval_data_file=$EXP_TEST_FILE \
    --line_by_line \
    --per_device_eval_batch_size=4 \
    --per_gpu_eval_batch_size=4 \
    --overwrite_output_dir \
    --save_file=$EXP_SAVE_NAME \
    --seed=$Seed

python ./train_t5.py \
    --output_dir=$OUTPUT_DIR \
    --model_type=t5 \
    --tokenizer_name=t5-large \
    --model_name_or_path=$GENERAL_MODEL_DIR\
    --do_eval \
    --eval_data_file=$GENERAL_TEST_FILE \
    --line_by_line \
    --per_device_eval_batch_size=4 \
    --per_gpu_eval_batch_size=4 \
    --overwrite_output_dir \
    --save_file=$GENERAL_SAVE_NAME  \
    --seed=$Seed

python ./train_t5.py \
    --output_dir=$OUTPUT_DIR \
    --model_type=t5 \
    --tokenizer_name=t5-large \
    --model_name_or_path=$AS_MODEL_DIR\
    --do_eval \
    --eval_data_file=$AS_TEST_FILE \
    --line_by_line \
    --per_device_eval_batch_size=4 \
    --per_gpu_eval_batch_size=4 \
    --overwrite_output_dir \
    --save_file=$AS_SAVE_NAME  \
    --seed=$Seed

python ./distill.py \
   --output_dir=$OUTPUT_DIR \
   --folder=$DATA_FOLDER \
   --extra_name=$EXTRA \
   --general_name=$TEST_FILE_NAME \
   --as_name=$WO_TEST_FILE_NAME \
   --exp_name=$EXP_VERIFIER_NAME \
   --general_model=$GENERAL_MODEL_NAME \
   --as_model=$AS_MODEL_NAME \
   --exp_model=$EXP_MODEL_NAME 


# Train with distilled data
python ./train_t5_with_today.py \
  --output_dir=$MODEL_OUTPUT_DIR \
  --model_type=t5 \
  --tokenizer_name=t5-large \
  --model_name_or_path=$MODEL_NAME \
  --do_train \
  --num_train_epochs=50\
  --max_steps=$Step\
  --train_data_file_1=$TRAIN_FILE_DISTILL \
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
