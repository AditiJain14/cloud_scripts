#!/usr/bin/bash

ROOT="/home/aditi/mma_runs"

DATA="${ROOT}/data/vi_en/data-bin"

EXP="${ROOT}/experiments/en_vi"

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
#     lambda=0.04 #$1
    split=$1
    ckpt_upper=$2
    ckpt_lower=$3

    name="lmloss_pretrainedwithadvi_0.1" #"latency_${lambda}"
    # name="single_path_latencyen-vi_0.4"
    # name="lmloss_latency_0.1_0.1en-viboostlm_0.2"
#     name="lmloss_pretraineden-vi_0.3"
    EXPT="${EXP}/infinite/${name}"
    RES="${EXPT}/results"
    mkdir -p "${RES}"

    modelfile="${EXPT}/checkpoints"

    # average last 5 checkpoints
    for ((i=$ckpt_upper; i>=$ckpt_lower; i--)); do
        bound=$i
        ckpt="avgmodel_${bound}.pt"

        echo "averaging checkpoints uptil $i"
        python "${FAIRSEQ}/scripts/average_checkpoints.py" --inputs ${modelfile} --num-epoch-checkpoints 3 --checkpoint-upper-bound $bound --output ${modelfile}/${ckpt} 

        #ckpt="checkpoint_best.pt"
        # ckpt="checkpoint40.pt"

        pred="pred_${bound}"
        mkdir -p "${RES}/${split}/${pred}"

        sleep 1

        # batched prediction
        echo "generating $split .."

        python "${FAIRSEQ}/fairseq_cli/generate.py" ${data} \
        --path "${modelfile}/${ckpt}" \
        --source-lang "en" --target-lang "vi" \
        --batch-size 50 \
        --beam 1 \
        --left-pad-source \
        --remove-bpe \
        --gen-subset ${split} \
        > "${RES}/${split}/${pred}/pred.out"

        grep ^H "${RES}/${split}/${pred}/pred.out" | cut -f1,3- | cut -c3- | sort -k1n | cut -f2- > "${RES}/${split}/${pred}/pred.translation"
        # sacremoses -l en -j 4 detokenize < "${RES}/${testk}/pred.translation" > "${RES}/${testk}/pred.translation.detok"

        grep ^T- "${RES}/${split}/${pred}/pred.out" | cut -f1,2- | cut -c3- | sort -k1n | cut -f2- > "${RES}/${split}/${pred}/pred.ref"
        # sacremoses -l en -j 4 detokenize < "${RES}/${testk}/pred.ref" > "${RES}/${testk}/pred.ref.detok"

        # write scores to a file for easy tracking
        tail -n 5 "${RES}/${split}/${pred}/pred.out" > "${RES}/${split}/${pred}/score_summary.txt"

        raw_ref="${DATA}/../raw/${split}.vi"
        # lower cased
        echo "lower-cased:--" >> "${RES}/${split}/${pred}/score_summary.txt" 
        perl "${FAIRSEQ}/multi-bleu.perl" -lc "${RES}/${split}/${pred}/pred.ref" < "${RES}/${split}/${pred}/pred.translation" >> "${RES}/${split}/${pred}/score_summary.txt" 
        # cased
        echo "cased:--" >> "${RES}/${split}/${pred}/score_summary.txt" 
        perl "${FAIRSEQ}/multi-bleu.perl" "${raw_ref}" < "${RES}/${split}/${pred}/pred.translation" >> "${RES}/${split}/${pred}/score_summary.txt" 
        
        # display scores
        cat "${RES}/${split}/${pred}/score_summary.txt"

        sleep 2

    done
    
    
}


##############
# generate_single_path [split] [upper] [lower]
generate_single_path valid 48 39
