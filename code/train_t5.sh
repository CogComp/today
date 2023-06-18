
export Seed=10
export Step=2000
export DATASET='combine' #matres #tracie
export NEW_MODEL_NAME="best_model_t5_reproduce"

export MODEL_NAME=t5-large
export OUTPUT_DIR="../model/${NEW_MODEL_NAME}"



# Relative-Label distilled GPT-3.5 Training set
export TRAIN_FILE_1="../data/gpt3.5/t5_iter0_distilled_gpt3.5_combined_today_train.txt"
# Relative-Label Original Today Training set / General Explanation Verifier
#export TRAIN_FILE_1="../data/today/today_train_with_explanation.txt"
# Additional Sentence Verifier
#export TRAIN_FILE_1="../data/today/today_train_without_explanation.txt"

# Hard-Label Training set
#If DATASET == 'combine'
export TRAIN_FILE_2="../data/combine/combined_train.txt"
#If DATASET == 'matres'
#export TRAIN_FILE_2="../data/matres/matres_train.txt"
#If DATASET == 'tracie'
#export TRAIN_FILE_2="../data/tracie/tracie_train.txt"




# Training
python ./train_t5_with_today.py \
   --output_dir=$OUTPUT_DIR \
   --model_type=t5 \
   --tokenizer_name=t5-large \
   --model_name_or_path=$MODEL_NAME \
   --do_train \
   --num_train_epochs=50\
   --max_steps=$Step\
   --train_data_file_1=$TRAIN_FILE_1 \
   --train_data_file_2=$TRAIN_FILE_2 \
   --line_by_line \
   --per_gpu_train_batch_size=2 \
   --per_device_train_batch_size=1 \
   --gradient_accumulation_steps=8 \
   --per_device_eval_batch_size=1 \
   --per_gpu_eval_batch_size=1 \
   --loss_alpha=10 \
   --loss_beta=2 \
   --save_steps=5000 \
   --logging_steps=1000 \
   --overwrite_output_dir \
   --seed=$Seed
