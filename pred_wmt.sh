
ROOT="/home/aditi/mma_runs"

DATA="${ROOT}/data/de_en/data-bin"

EXP="${ROOT}/experiments/de_en"

###############
# name="base"
# testk=3
export CUDA_VISIBLE_DEVICES=0
###############

# EXPT="${ROOT}/experiments/iwslt14_ende/${name}"

# RES="${EXPT}/results/"
# mkdir -p "${RES}/${testk}"

FAIRSEQ="${ROOT}/mma"

data="${DATA}"
# modelfile="${EXPT}/checkpoints"

generate_single_path(){
    lambda=0.04 #$1

    name="lmlossApril11_0.1_0.3" #"latency_${lambda}"

    EXPT="${EXP}/infinite/${name}"
    RES="${EXPT}/results/action"
    mkdir -p "${RES}"

    modelfile="${EXPT}/checkpoints"

    # average last 5 checkpoints
    python "${FAIRSEQ}/scripts/average_checkpoints.py" --inputs ${modelfile} --num-epoch-checkpoints 5 --checkpoint-upper-bound 33 --output ${modelfile}/average-model.pt 

    # bsz 1 prediction
    # python generate.py ${DATA} --path $modelfile/average-model.pt --batch-size 1 --beam 1 --left-pad-source False --fp16  --remove-bpe --test-wait-k ${testk} --sim-decoding > pred.out

#     ckpt=avgmodel_31.pt
#     ckpt="checkpoint_best.pt"
#       ckpt="checkpoint_last.pt"
      ckpt="average-model.pt"
    # ckpt="checkpoint40.pt"

    # batched prediction
    python "${FAIRSEQ}/fairseq_cli/generate.py" ${data} \
    --path "${modelfile}/${ckpt}" \
    --source-lang "de" --target-lang "en" \
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

    raw_ref="${DATA}/../raw/test.en"
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
