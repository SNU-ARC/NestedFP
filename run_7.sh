#bash scripts/acc_eval.sh 6 /home/ubuntu/models/Mistral-Small-24B-Base-2501 bbh_zeroshot         1 --nestedfp
#bash scripts/acc_eval.sh 6 /home/ubuntu/models/Mistral-Small-24B-Base-2501 minerva_math         1 --nestedfp
#bash scripts/acc_eval.sh 6 /home/ubuntu/models/Mistral-Small-24B-Base-2501 leaderboard_mmlu_pro 1 --nestedfp

#bash scripts/acc_eval.sh 0 /home/ubuntu/models/Mistral-Small-24B-Base-2501 bbh_zeroshot         1 &
#bash scripts/acc_eval.sh 1 /home/ubuntu/models/Mistral-Small-24B-Base-2501 minerva_math         1 &
bash scripts/acc_eval.sh 7 /home/ubuntu/models/Mistral-Small-24B-Base-2501 leaderboard_mmlu_pro 1
