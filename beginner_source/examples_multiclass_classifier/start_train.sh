if [ ! -d ".log" ]; then
    mkdir .log
fi

CUDA_VISIBLE_DEVICES=0 python multiclass_text_classifier.py > .log/$(date '+%m%d_%H%M%S').log
