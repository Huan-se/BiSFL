dir="./data"
MAX_JOBS=2

for file in "$dir"/*
do
    name=$(basename "$file")
    label="${name%.*}"

    python run_IG_test.py --image_path "$file" --label "$label" > log_$label.txt 2>&1 &

    if [[ $(jobs -r -p | wc -l) -ge $MAX_JOBS ]]; then
        wait -n
    fi

done

wait