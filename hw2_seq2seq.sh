wget -O "models/hw2_special.ckpt.data-00000-of-00001" "https://www.dropbox.com/s/i8c0jerz3xd445p/hw2_special.ckpt.data-00000-of-00001?dl=0"
python hw2_seq2seq.py "$1" "$2"
