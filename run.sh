bash scripts/acc_eval.sh 0,1,2,3,4,5,6,7 /home/ubuntu/models/Llama-3.1-70B bbh_zeroshot         1
bash scripts/acc_eval.sh 0,1,2,3,4,5,6,7 /home/ubuntu/models/Llama-3.1-70B minerva_math         1

bash scripts/acc_eval.sh 0,1,2,3,4,5,6,7 /home/ubuntu/models/DeepSeek-R1-Distill-Llama-70B  bbh_zeroshot         1
bash scripts/acc_eval.sh 0,1,2,3,4,5,6,7 /home/ubuntu/models/DeepSeek-R1-Distill-Llama-70B  minerva_math         1

#bash scripts/acc_eval.sh 0,1,2,3,4,5,6,7 /home/ubuntu/models/Llama-3.1-70B leaderboard_mmlu_pro 1
#bash scripts/acc_eval.sh 0,1,2,3,4,5,6,7 /home/ubuntu/models/DeepSeek-R1-Distill-Llama-70B  leaderboard_mmlu_pro 1


"""
bash scripts/acc_eval.sh 0,1,2,3,4,5,6,7 /home/ubuntu/models/Llama-3.1-70B bbh_zeroshot         1 --nestedfp
bash scripts/acc_eval.sh 0,1,2,3,4,5,6,7 /home/ubuntu/models/Llama-3.1-70B minerva_math         1 --nestedfp
bash scripts/acc_eval.sh 0,1,2,3,4,5,6,7 /home/ubuntu/models/Llama-3.1-70B leaderboard_mmlu_pro 1 --nestedfp

bash scripts/acc_eval.sh 0,1,2,3,4,5,6,7 /home/ubuntu/models/DeepSeek-R1-Distill-Llama-70B  bbh_zeroshot         1 --nestedfp
bash scripts/acc_eval.sh 0,1,2,3,4,5,6,7 /home/ubuntu/models/DeepSeek-R1-Distill-Llama-70B  minerva_math         1 --nestedfp
bash scripts/acc_eval.sh 0,1,2,3,4,5,6,7 /home/ubuntu/models/DeepSeek-R1-Distill-Llama-70B  leaderboard_mmlu_pro 1 --nestedfp

./run_1.sh &
./run_2.sh &
./run_3.sh &
./run_4.sh &
./run_5.sh &
./run_6.sh &
./run_7.sh &
./run_8.sh &

wait

bash scripts/acc_eval.sh 0,1,2,3,4,5,6,7 /home/ubuntu/models/Llama-3.1-70B-FP8-Dynamic-Half    bbh_zeroshot         1 
bash scripts/acc_eval.sh 0,1,2,3,4,5,6,7 /home/ubuntu/models/Llama-3.1-70B-FP8-Dynamic-Half    minerva_math         1 
bash scripts/acc_eval.sh 0,1,2,3,4,5,6,7 /home/ubuntu/models/Llama-3.1-70B-FP8-Dynamic-Half    leaderboard_mmlu_pro 1 

bash scripts/acc_eval.sh 0,1,2,3,4,5,6,7 /home/ubuntu/models/DeepSeek-R1-Distill-Llama-70B-FP8-Dynamic-Half   bbh_zeroshot         1 
bash scripts/acc_eval.sh 0,1,2,3,4,5,6,7 /home/ubuntu/models/DeepSeek-R1-Distill-Llama-70B-FP8-Dynamic-Half   minerva_math         1 
bash scripts/acc_eval.sh 0,1,2,3,4,5,6,7 /home/ubuntu/models/DeepSeek-R1-Distill-Llama-70B-FP8-Dynamic-Half   leaderboard_mmlu_pro 1 

wait
"""