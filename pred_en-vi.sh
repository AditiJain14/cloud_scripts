
ROOT="/home/aditi/mma_runs"

DATA="${ROOT}/data/vi_en/data-bin"

EXP="${ROOT}/experiments/en_vi"

###############
# name="base"
# testk=3
export CUDA_VISIBLE_DEVICES=1
###############

# EXPT="${ROOT}/experiments/iwslt14_ende/${name}"

# RES="${EXPT}/results/"
# mkdir -p "${RES}/${testk}"

FAIRSEQ="${ROOT}/mma"

data="${DATA}"
# modelfile="${EXPT}/checkpoints"

generate_single_path(){
    lambda=0.04 #$1

    name="lmloss_latency_0.1_0.3_withchkpt0.3" #"latency_${lambda}"
    # name="single_path_latencyen-vi_0.4"
    # name="lmloss_latency_0.1_0.1en-viboostlm_0.2"

    EXPT="${EXP}/infinite/${name}"
    RES="${EXPT}/results"
    mkdir -p "${RES}"

    modelfile="${EXPT}/checkpoints"

    # average last 5 checkpoints
    # python "${FAIRSEQ}/scripts/average_checkpoints.py" --inputs ${modelfile} --num-epoch-checkpoints 3 --checkpoint-upper-bound 37 --output ${modelfile}/average-model.pt 

    # bsz 1 prediction
    # python generate.py ${DATA} --path $modelfile/average-model.pt --batch-size 1 --beam 1 --left-pad-source False --fp16  --remove-bpe --test-wait-k ${testk} --sim-decoding > pred.out

    # ckpt=average-model.pt
    #ckpt="checkpoint_best.pt"
    ckpt="avgmodel_48.pt"

    # batched prediction
    python "${FAIRSEQ}/fairseq_cli/generate.py" ${data} \
    --path "${modelfile}/${ckpt}" \
    --source-lang "en" --target-lang "vi" \
    --batch-size 50 \
    --beam 1 \
    --left-pad-source \
    --remove-bpe \
    > "${RES}/pred.out"

    grep ^H "${RES}/pred.out" | cut -f1,3- | cut -c3- | sort -k1n | cut -f2- > "${RES}/pred.translation"
    # sacremoses -l en -j 4 detokenize < "${RES}/${testk}/pred.translation" > "${RES}/${testk}/pred.translation.detok"

    grep ^T- "${RES}/pred.out" | cut -f1,2- | cut -c3- | sort -k1n | cut -f2- > "${RES}/pred.ref"
    # sacremoses -l en -j 4 detokenize < "${RES}/${testk}/pred.ref" > "${RES}/${testk}/pred.ref.detok"

    # write scores to a file for easy tracking
    tail -n 5 "${RES}/pred.out" > "${RES}/score_summary.txt"

    raw_ref="${DATA}/../raw/test.vi"
    # lower cased
    echo "lower-cased:--" >> "${RES}/score_summary.txt" 
    perl "${FAIRSEQ}/multi-bleu.perl" -lc "${RES}/pred.ref" < "${RES}/pred.translation" >> "${RES}/score_summary.txt" 
    # cased
    echo "cased:--" >> "${RES}/score_summary.txt" 
    perl "${FAIRSEQ}/multi-bleu.perl" "${raw_ref}" < "${RES}/pred.translation" >> "${RES}/score_summary.txt" 
    
    # display scores
    cat "${RES}/score_summary.txt"
}

##############
generate_single_path
