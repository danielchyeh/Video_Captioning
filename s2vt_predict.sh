wget -O "models/model.ckpt.data-00000-of-00001" "https://www.dropbox.com/s/heqn54rpbyhuj5e/model.ckpt.data-00000-of-00001?dl=0"
python3 s2vt_predict.py "$1" "$2"
