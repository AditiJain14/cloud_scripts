
ROOT="/home/aditi/mma_runs"
# ROOT="path/to/working/dir"

DATA="${ROOT}/data/vi_en/data-bin"

EXPT="${ROOT}/experiments/en_vi"
mkdir -p ${EXPT}

FAIRSEQ="${ROOT}/mma"

USR="./examples/simultaneous_translation"




export CUDA_VISIBLE_DEVICES=0,1

# infinite lookback
mma_il(){
    lambda=$1
    name="single_path_latencyen-vi_${lambda}"
    #name="trained_lm${lambda}"
    export WANDB_NAME="${name}"

    CKPT="${EXPT}/infinite/${name}/checkpoints"
    TBOARD="${EXPT}/infinite/${name}/logs"
    mkdir -p ${CKPT} ${TBOARD}

    python ${FAIRSEQ}/train.py  --ddp-backend=no_c10d ${DATA} \
    --source-lang en --target-lang vi \
    --log-format simple --log-interval 50 \
    --arch transformer_monotonic_iwslt_de_en \
    --user-dir "${USR}" \
    --simul-type infinite_lookback \
    --mass-preservation \
    --optimizer adam \
    --adam-betas '(0.9, 0.98)' \
    --clip-norm 0.0 \
    --lr 5e-4 \
    --weight-decay 0.0001 \
    --lr-scheduler 'inverse_sqrt' \
    --warmup-init-lr 1e-07 \
    --warmup-updates 4000 \
    --dropout 0.3 \
    --criterion latency_augmented_label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --encoder-attention-heads 4 \
    --decoder-attention-heads 4 \
    --max-update 100000 \
    --latency-weight-avg  ${lambda} \
    --noise-var 1.5 \
    --left-pad-source \
    --single-path \
    --dual-weight 0.0 \
    --save-dir $CKPT \
    --max-tokens 9000 --update-freq 3 \
    --best-checkpoint-metric "ppl" \
    --max-epoch 43  --keep-last-epochs 12\
    --tensorboard-logdir ${TBOARD} --wandb-project LM_Adaptive_EnVi \
    | tee -a ${TBOARD}/train_log.txt

}
#Single Path on one gpu of nll22 
mma_il_with_pretrained(){
    lambda=$1
    name="single_path_latency_trial${lambda}"
    #name="trained_lm${lambda}"
    export WANDB_NAME="${name}"

    CKPT="${EXPT}/infinite/${name}/checkpoints"
    TBOARD="${EXPT}/infinite/${name}/logs"
    mkdir -p ${CKPT} ${TBOARD}

    python ${FAIRSEQ}/train.py  --ddp-backend=no_c10d ${DATA} \
    --source-lang vi --target-lang en \
    --log-format simple --log-interval 50 \
    --arch transformer_monotonic_iwslt_de_en \
    --user-dir "${USR}" \
    --simul-type infinite_lookback \
    --mass-preservation \
    --optimizer adam \
    --adam-betas '(0.9, 0.98)' \
    --clip-norm 0.0 \
    --lr 5e-4 \
    --weight-decay 0.0001 \
    --lr-scheduler 'inverse_sqrt' \
    --warmup-init-lr 1e-07 \
    --warmup-updates 4000 \
    --dropout 0.3 \
    --criterion latency_augmented_label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --encoder-attention-heads 4 \
    --decoder-attention-heads 4 \
    --max-update 100000 \
    --latency-weight-avg  ${lambda} \
    --noise-var 1.5 \
    --left-pad-source \
    --single-path \
    --dual-weight 0.0 \
    --save-dir $CKPT \
    --max-tokens 9000 --update-freq 3 \
    --best-checkpoint-metric "ppl" \
    --max-epoch 50  --keep-last-epochs 12\
    --restore-file "/cs/natlang-expts/aditi/mma_runs/experiments/vi_en/infinite/trained_lm0/checkpoints/checkpoint7.pt" \
    --tensorboard-logdir ${TBOARD} --wandb-project LM_Adaptive \
    | tee -a ${TBOARD}/train_log.txt

}
#   --keep-last-epochs 12
#mma-il with lm with 2 gpu 
mma_il_lm(){
    lambda=$1
    # name="single_path_latency_${lambda}"
    name="lmloss_latency_0.0_0.45_${lambda}"
    export WANDB_NAME="${name}"

    CKPT="${EXPT}/infinite/${name}/checkpoints"
    TBOARD="${EXPT}/infinite/${name}/logs"
    mkdir -p ${CKPT} ${TBOARD}

    python ${FAIRSEQ}/train.py  --ddp-backend=no_c10d ${DATA} \
    --source-lang en --target-lang vi \
    --log-format simple --log-interval 50 \
    --arch transformer_monotonic_iwslt_de_en \
    --user-dir "${USR}" \
    --simul-type infinite_lookback \
    --mass-preservation \
    --optimizer adam \
    --adam-betas '(0.9, 0.98)' \
    --clip-norm 0.0 \
    --lr 5e-4 \
    --weight-decay 0.0001 \
    --lr-scheduler 'inverse_sqrt' \
    --warmup-init-lr 1e-07 \
    --warmup-updates 4000 \
    --dropout 0.3 \
    --criterion latency_augmented_label_smoothed_cross_entropy_cbmi \
    --label-smoothing 0.1 \
    --encoder-attention-heads 4 \
    --decoder-attention-heads 4 \
    --max-update 100000 \
    --latency-weight-avg  ${lambda} \
    --noise-var 1.5 \
    --left-pad-source \
    --single-path \
    --dual-weight 0.0 \
    --save-dir $CKPT \
    --max-tokens 3600 --update-freq 2 \
    --best-checkpoint-metric "ppl" \
    --keep-last-epochs 15 \
    --add-language-model \
    --share-lm-decoder-softmax-embed \
    --pretrain-steps 3000 \
    --token-scale 0.0 --sentence-scale 0.45 \
    --wandb-project LM_Adaptive_EnVi \
    --empty-cache-freq 45 --max-epoch 52\
    | tee -a ${TBOARD}/train_log.txt
    # --tensorboard-logdir ${TBOARD} \
    #dont use cbmi loss for getting checkpoints for lambda>0.1, set pretrain-steps high. 
    #This will also train LM decoder with rate lm_rate*10
}

#loading a LM Loss model with checkpoint
mma_il_lm_from_chkpt(){
    lambda=$1
    file=$2
    # name="single_path_latency_${lambda}"
    name="lmloss_latency0_0.0_0.3_withchkpt${lambda}"
    export WANDB_NAME="${name}"

    CKPT="${EXPT}/infinite/${name}/checkpoints"
    TBOARD="${EXPT}/infinite/${name}/logs"
    mkdir -p ${CKPT} ${TBOARD}

    python ${FAIRSEQ}/train.py  --ddp-backend=no_c10d ${DATA} \
    --source-lang en --target-lang vi \
    --log-format simple --log-interval 50 \
    --arch transformer_monotonic_iwslt_de_en \
    --user-dir "${USR}" \
    --simul-type infinite_lookback \
    --mass-preservation \
    --optimizer adam \
    --adam-betas '(0.9, 0.98)' \
    --clip-norm 0.0 \
    --lr 5e-4 \
    --weight-decay 0.0001 \
    --lr-scheduler 'inverse_sqrt' \
    --warmup-init-lr 1e-07 \
    --warmup-updates 4000 \
    --dropout 0.3 \
    --criterion latency_augmented_label_smoothed_cross_entropy_cbmi \
    --label-smoothing 0.1 \
    --encoder-attention-heads 4 \
    --decoder-attention-heads 4 \
    --max-update 100000 \
    --latency-weight-avg  ${lambda} \
    --noise-var 1.5 \
    --left-pad-source \
    --single-path \
    --dual-weight 0.0 \
    --save-dir $CKPT \
    --max-tokens 3600 --update-freq 2 \
    --best-checkpoint-metric "ppl" \
    --keep-last-epochs 20 \
    --add-language-model \
    --share-lm-decoder-softmax-embed \
    --pretrain-steps 0 \
    --token-scale 0.0 --sentence-scale 0.3 \
    --wandb-project LM_Adaptive_EnVi \
    --restore-file $file \
    --empty-cache-freq 45 --max-epoch 58\
    | tee -a ${TBOARD}/train_log.txt
    # --tensorboard-logdir ${TBOARD} \

}


wait_info_adaptive_train(){
    name="expt_adapt_train_T1.75"
    CKPT="${EXPT}/${name}/checkpoints"
    TBOARD="${EXPT}/${name}/logs"
    mkdir -p ${CKPT} ${TBOARD}

    TGT_DICT="${DATA}/dict.en.txt"

    echo "Training Wait-Info + Adaptive .."

    python3 ${FAIRSEQ}/train.py --ddp-backend=no_c10d ${DATA} --arch transformer_iwslt_de_en \
    --source-lang de --target-lang en \
    --share-decoder-input-output-embed \
    --optimizer adam \
    --adam-betas '(0.9, 0.98)' \
    --clip-norm 0.0 \
    --log-format simple \
    --log-interval 100 \
    --dropout 0.3 \
    --encoder-attention-heads 8 \
    --decoder-attention-heads 8 \
    --criterion label_smoothed_cross_entropy_exponential_adapt \
    --lr-scheduler inverse_sqrt \
    --lr 5e-4 \
    --warmup-init-lr 1e-07 \
    --warmup-updates 4000 \
    --adaptive-training \
    --dict-file ${TGT_DICT} \
    --adaptive-method 'exp' \
    --adaptive-T 1.75 \
    --weight-drop 0.3 \
    --weight-decay 0.0 \
    --label-smoothing 0.1 \
    --left-pad-source False \
    --save-dir ${CKPT} \
    --max-tokens 8192 --update-freq 1 \
    --max-update 60000 \
    --fp16 \
    --keep-last-epochs 10 \
    --best-checkpoint-metric "ppl" \
    --patience 10 \
    --tensorboard-logdir ${TBOARD} 
    #| tee -a ${TBOARD}/train_log.txt

}

wait_info_adaptive_ft(){
    RESTORE="${EXPT}/base/checkpoints/checkpoint30.pt"

    CKPT="${EXPT}/expt_adapt_0.1/checkpoints"
    TBOARD="${EXPT}/expt_adapt_0.1/logs"
    mkdir -p ${CKPT} ${TBOARD}

    TGT_DICT="${DATA}/dict.en.txt"

    echo "Fine-tuning Wait-Info + Adaptive .."

    python3 ${FAIRSEQ}/train.py --ddp-backend=no_c10d ${DATA} --arch transformer_iwslt_de_en \
    --source-lang de --target-lang en \
    --share-decoder-input-output-embed \
    --optimizer adam \
    --adam-betas '(0.9, 0.98)' \
    --clip-norm 0.0 \
    --log-format simple \
    --log-interval 100 \
    --dropout 0.3 \
    --encoder-attention-heads 8 \
    --decoder-attention-heads 8 \
    --criterion label_smoothed_cross_entropy_exponential_adapt \
    --lr-scheduler inverse_sqrt \
    --lr 0.00028 \
    --warmup-init-lr 1e-07 \
    --warmup-updates 1000 \
    --reset-optimizer --reset-lr-scheduler \
    --adaptive-training \
    --dict-file ${TGT_DICT} \
    --adaptive-method 'exp' \
    --adaptive-T 1.75 \
    --weight-drop 0.1 \
    --weight-decay 0.0 \
    --label-smoothing 0.1 \
    --left-pad-source False \
    --restore-file ${RESTORE} \
    --save-dir ${CKPT} \
    --max-tokens 8192 --update-freq 1 \
    --max-update 15000 \
    --fp16 \
    --save-interval-updates 1000 \
    --keep-interval-updates 2 \
    --tensorboard-logdir ${TBOARD} 
    #| tee -a ${TBOARD}/train_log.txt

}
#LM Loss with frozen pretrained LM 
mma_il_freezelmchkpt(){
    lambda=$1
    name="lmloss_pretrainedfinal50_${lambda}"
    export WANDB_NAME="${name}"
    CKPT="${EXPT}/infinite/${name}/checkpoints"
    TBOARD="${EXPT}/infinite/${name}/logs"
    pre_path="/cs/natlang-expts/aditi/mma_runs/experiments/vi_en/infinite/lmloss_pretrainedtrial_0/checkpoints/checkpoint7.pt"
    #pre_path="/cs/natlang-expts/aditi/mma_runs/experiments/vi_en/infinite/trained_lm/checkpoints/checkpoint60.pt"
    mkdir -p ${CKPT} ${TBOARD}

    python ${FAIRSEQ}/train.py  --ddp-backend=no_c10d ${DATA} \
    --source-lang vi --target-lang en \
    --log-format simple --log-interval 50 \
    --arch transformer_monotonic_iwslt_de_en \
    --user-dir "${USR}" \
    --simul-type infinite_lookback \
    --mass-preservation \
    --optimizer adam \
    --adam-betas '(0.9, 0.98)' \
    --clip-norm 0.0 \
    --lr 5e-4 \
    --weight-decay 0.0001 \
    --lr-scheduler 'inverse_sqrt' \
    --warmup-init-lr 1e-07 \
    --warmup-updates 4000 \
    --dropout 0.3 \
    --criterion latency_augmented_label_smoothed_cross_entropy_cbmi \
    --label-smoothing 0.1 \
    --encoder-attention-heads 4 \
    --decoder-attention-heads 4 \
    --max-update 100000 \
    --latency-weight-avg  ${lambda} \
    --noise-var 1.5 \
    --left-pad-source \
    --single-path \
    --dual-weight 0.0 \
    --save-dir $CKPT \
    --max-tokens 6750 --update-freq 2 \
    --best-checkpoint-metric "ppl" \
    --add-language-model \
    --share-lm-decoder-softmax-embed \
    --use-pretrained-lm \
    --token-scale 0.1 --sentence-scale 0.1 \
    --empty-cache-freq 45 \
    --keep-last-epochs 12\
    --pretrain-steps 0 --max-epoch 40\
    --wandb-project LM_Adaptive \
    --freeze-pretrained-lm \
    --pretrained-lm-path $pre_path\
    | tee -a ${TBOARD}/train_log.txt
    # --tensorboard-logdir ${TBOARD} \
    #--finetune-fix-lm True \
    #--freeze-pretrained-lm 
    #--pretrained-lm-path $pre_path \
    #--restore-file "/cs/natlang-expts/aditi/mma_runs/experiments/vi_en/infinite/trained_lm0/checkpoints/checkpoint7.pt" \
#--keep-last-epochs 12 \

}


train_lm_only(){
    lambda=$1
    name="trained_lm_lmloss"
    export WANDB_NAME="${name}"

    CKPT="${EXPT}/infinite/${name}/checkpoints"
    TBOARD="${EXPT}/infinite/${name}/logs"
    mkdir -p ${CKPT} ${TBOARD}

    python ${FAIRSEQ}/train.py  --ddp-backend=no_c10d ${DATA} \
    --source-lang vi --target-lang en \
    --log-format simple --log-interval 50 \
    --arch transformer_monotonic_iwslt_de_en \
    --user-dir "${USR}" \
    --simul-type infinite_lookback \
    --mass-preservation \
    --optimizer adam \
    --adam-betas '(0.9, 0.98)' \
    --clip-norm 0.0 \
    --lr 5e-4 \
    --weight-decay 0.0001 \
    --lr-scheduler 'inverse_sqrt' \
    --warmup-init-lr 1e-07 \
    --warmup-updates 4000 \
    --dropout 0.3 \
    --criterion latency_augmented_label_smoothed_cross_entropy_cbmi \
    --label-smoothing 0.1 \
    --encoder-attention-heads 1 \
    --encoder-layers 1 \
    --decoder-attention-heads 4 \
    --max-update 100000 \
    --latency-weight-avg  ${lambda} \
    --noise-var 1.5 \
    --left-pad-source \
    --single-path \
    --dual-weight 0.0 \
    --save-dir $CKPT \
    --max-tokens 4500 --update-freq 3 \
    --add-language-model \
    --share-lm-decoder-softmax-embed \
    --token-scale 0.1 --sentence-scale 0.1 \
    --empty-cache-freq 45 \
    --train-only-lm --disable-validation \
    --wandb-project Lm_Adaptive 
}
#--use-pretrained-lm \
#--keep-last-epochs 12 \

  #--pretrained-lm-path "/cs/natlang-expts/aditi/mma_runs/experiments/vi_en/infinite/lmloss_latency_0.1_0.1_0.04/checkpoints/checkpoint_best.pt" \
###############################################
export CUDA_VISIBLE_DEVICES=0,1,2,3

# mma_il 0.05
# mma_h
# mma_wait_k
# wait_info
# wait_info_adaptive_ft

# wait_info_adaptive_train

# mma_il_lm 0.1
# mma_il_lm_pre 0.3
# mma_il_lm_only 0

mma_il_lm_from_chkpt 0 "/home/aditi/mma_runs/experiments/en_vi/infinite/lmloss_latency_0.0_0.45_0.1/checkpoints/checkpoint45.pt"
