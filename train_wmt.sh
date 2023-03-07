
ROOT="/home/aditi/mma_runs"
# ROOT="path/to/working/dir"

DATA="${ROOT}/data/de_en/data-bin"

EXPT="${ROOT}/experiments/de_en"
mkdir -p ${EXPT}

FAIRSEQ="${ROOT}/mma"

USR="./examples/simultaneous_translation"




export CUDA_VISIBLE_DEVICES=0,1

# infinite lookback
#Single Path 
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
    --latency-weight-avg ${lambda} \
    --noise-var 1.5 \
    --left-pad-source \
    --single-path \
    --dual-weight 0.0 \
    --save-dir $CKPT \
    --max-tokens 3375 --update-freq 4 \
    --best-checkpoint-metric "ppl" \
    --max-epoch 50 --keep-last-epochs 15\
    --tensorboard-logdir ${TBOARD} \
    --wandb-project LM_Adaptive_EnVi \
    --restore-file "/cs/natlang-expts/aditi/mma_runs/experiments/vi_en/infinite/single_path_latencyen-vi_0.3/checkpoints/checkpoint37.pt"\
    | tee -a ${TBOARD}/train_log.txt
}
#Single Path on one gpu of nll22 
mma_il_with_pretrained(){
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
    --latency-weight-avg ${lambda} \
    --noise-var 1.5 \
    --left-pad-source \
    --single-path \
    --dual-weight 0.0 \
    --save-dir $CKPT \
    --max-tokens 3375 --update-freq 4 \
    --best-checkpoint-metric "ppl" \
    --max-epoch 52  --keep-last-epochs 15\
    --tensorboard-logdir ${TBOARD} --wandb-project LM_Adaptive_EnVi \
    --restore-file "/cs/natlang-expts/aditi/mma_runs/experiments/vi_en/infinite/single_path_latencyen-vi_0/checkpoints/checkpoint8.pt" \
| tee -a ${TBOARD}/train_log.txt
}
#   --keep-last-epochs 12

mma_il_lm(){
    lambda=$1
    # name="single_path_latency_${lambda}"
    name="lmloss_latency_0.1_0.1_fp16${lambda}"
    export WANDB_NAME="${name}"

    CKPT="${EXPT}/infinite/${name}/checkpoints"
    TBOARD="${EXPT}/infinite/${name}/logs"
    mkdir -p ${CKPT} ${TBOARD}

    python ${FAIRSEQ}/train.py  --ddp-backend=no_c10d ${DATA} \
    --source-lang de --target-lang en \
    --log-format simple --log-interval 100 \
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
    --encoder-attention-heads 8 \
    --decoder-attention-heads 8 \
    --max-update 100000 \
    --latency-weight-avg  ${lambda} \
    --noise-var 1.5 \
    --left-pad-source \
    --single-path \
    --dual-weight 0.0 \
    --save-dir $CKPT \
    --max-tokens 5000 --update-freq 2 \
    --best-checkpoint-metric "ppl" \
    --add-language-model\
    --share-lm-decoder-softmax-embed \
    --pretrain-steps 12000 --keep-last-epochs 12\
    --token-scale 0.1 --sentence-scale 0.1\
    --wandb-project LM_Adaptive_DeEn --fp16 --reset-optimizer\
    --empty-cache-freq 45 --max-epoch 30\
    | tee -a ${TBOARD}/train_log.txt
    # --tensorboard-logdir ${TBOARD} \
        # --keep-last-epochs 20 \
    #dont use cbmi loss for getting checkpoints for lambda>0.1, set pretrain-steps high. 
    #This will also train LM decoder with rate lm_rate*10
    #load this checkpoint for lambda>0.1 runs 
    #--restore-file ""\

}

#loading a LM Loss model with checkpoint
mma_il_lm_from_chkpt(){
    lambda=$1
    # name="single_path_latency_${lambda}"
    name="lmloss_pretraineden-vi_${lambda}"
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
    --max-tokens 3375 --update-freq 4 \
    --best-checkpoint-metric "ppl" \
    --keep-last-epochs 15 \
    --add-language-model \
    --share-lm-decoder-softmax-embed \
    --pretrain-steps 3000 \
    --token-scale 0.1 --sentence-scale 0.1 \
    --wandb-project LM_Adaptive_EnVi \
    --restore-file "/cs/natlang-expts/aditi/mma_runs/experiments/vi_en/infinite/lmloss_pretraineden-vi_0/checkpoints/checkpoint12.pt" \
    --empty-cache-freq 45 --max-epoch 52\
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
    name="frozenlm_pretraineden-vi_${lambda}"
    export WANDB_NAME="${name}"
    CKPT="${EXPT}/infinite/${name}/checkpoints"
    TBOARD="${EXPT}/infinite/${name}/logs"
    pre_path="/cs/natlang-expts/aditi/mma_runs/experiments/en_vi/infinite/lmloss_pretraineden-vi_0.1_0.3_0.4/checkpoints/checkpoint_last.pt"
    #pre_path="/cs/natlang-expts/aditi/mma_runs/experiments/vi_en/infinite/trained_lm/checkpoints/checkpoint60.pt"
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
    --max-tokens 3375 --update-freq 4 \
    --best-checkpoint-metric "ppl" \
    --add-language-model \
    --share-lm-decoder-softmax-embed \
    --use-pretrained-lm \
    --token-scale 0.1 --sentence-scale 0.1 \
    --empty-cache-freq 45 \
    --keep-last-epochs 20\
    --pretrain-steps 4000 --max-epoch 20\
    --wandb-project LM_Adaptive_EnVi \
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
    name="trained_lm_envi"
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
    --wandb-project Lm_Adaptive_EnVi --max-epoch 40\ 
}
#--use-pretrained-lm \
#--keep-last-epochs 12 \

  #--pretrained-lm-path "/cs/natlang-expts/aditi/mma_runs/experiments/vi_en/infinite/lmloss_latency_0.1_0.1_0.04/checkpoints/checkpoint_best.pt" \
###############################################
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# mma_il 0.3
# train_lm_only 0
# mma_h
# mma_wait_k
# wait_info
# wait_info_adaptive_ft
# mma_il_freezelmchkpt 0
# wait_info_adaptive_train
# mma_il_with_pretrained 0.4
mma_il_lm 0.25
# mma_il_lm_pre 0.4
# mma_il_lm_only 0
# mma_il_lm_from_chkpt 0.4  
