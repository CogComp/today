# You only need to enter the model name
export INPUT_NAME="best_model_checkpoint"
export MODEL_NAME="../model/${INPUT_NAME}" 

# Evaluation set
export OUTPUT_DIR="../experiment_result/${INPUT_NAME}"

export TEST_FILE_1="../data/today/today_test_with_explanation_gold.txt"
export TEST_FILE_2="../data/today/today_test_without_explanation.txt"
export TEST_FILE_3="../data/tracie/tracie_test.txt"
export TEST_FILE_4="../data/matres/matres_test.txt"
export SAVE_NAME_1="${INPUT_NAME}_today_test_with_explanation_gold"
export SAVE_NAME_2="${INPUT_NAME}_today_test_without_explanation"
export SAVE_NAME_3="${INPUT_NAME}_tracie_test"
export SAVE_NAME_4="${INPUT_NAME}_matres_test"


#Evaluation
python ./train_t5.py \
    --output_dir=$OUTPUT_DIR \
    --model_type=t5 \
    --tokenizer_name=t5-large \
    --model_name_or_path=$MODEL_NAME\
    --do_eval \
    --eval_data_file=$TEST_FILE_1 \
    --line_by_line \
    --per_device_eval_batch_size=4 \
    --per_gpu_eval_batch_size=4 \
    --overwrite_output_dir \
    --save_file=$SAVE_NAME_1 \
    --seed=10

python ./train_t5.py \
    --output_dir=$OUTPUT_DIR \
    --model_type=t5 \
    --tokenizer_name=t5-large \
    --model_name_or_path=$MODEL_NAME\
    --do_eval \
    --eval_data_file=$TEST_FILE_2 \
    --line_by_line \
    --per_device_eval_batch_size=4 \
    --per_gpu_eval_batch_size=4 \
    --overwrite_output_dir \
    --save_file=$SAVE_NAME_2 \
    --seed=10

python ./train_t5.py \
    --output_dir=$OUTPUT_DIR \
    --model_type=t5 \
    --tokenizer_name=t5-large \
    --model_name_or_path=$MODEL_NAME\
    --do_eval \
    --eval_data_file=$TEST_FILE_3 \
    --line_by_line \
    --per_device_eval_batch_size=4 \
    --per_gpu_eval_batch_size=4 \
    --overwrite_output_dir \
    --save_file=$SAVE_NAME_3 \
    --seed=10

python ./train_t5.py \
    --output_dir=$OUTPUT_DIR \
    --model_type=t5 \
    --tokenizer_name=t5-large \
    --model_name_or_path=$MODEL_NAME\
    --do_eval \
    --eval_data_file=$TEST_FILE_4 \
    --line_by_line \
    --per_device_eval_batch_size=4 \
    --per_gpu_eval_batch_size=4 \
    --overwrite_output_dir \
    --save_file=$SAVE_NAME_4 \
    --seed=10


python ./evaluator_today.py \
    --test_file=$TEST_FILE_1 \
    --output_dir=$OUTPUT_DIR \
    --name=$SAVE_NAME_1

python ./evaluator_today.py \
    --test_file=$TEST_FILE_2 \
    --output_dir=$OUTPUT_DIR \
    --name=$SAVE_NAME_2

python ./evaluator_origin.py \
    --test_file=$TEST_FILE_3 \
    --output_dir=$OUTPUT_DIR \
    --name=$SAVE_NAME_3

python ./evaluator_origin.py \
    --test_file=$TEST_FILE_4 \
    --output_dir=$OUTPUT_DIR \
    --name=$SAVE_NAME_4