# Video-Captioning
MLDS2017 Project2
## Quick start
1. Download MSVD dataset which provides features extracted by VGG that is pretrained on the ImageNet.
Link: https://drive.google.com/file/d/0B18IKlS3niGFNlBoaHJTY3NXUkE/view (provided by MLDS2017) and put MSVD dataset under video-captioning folder

2. Create a .txt file and name it "sample_output_testset.txt". (Actually we have peer review section for MLDS2017, so if you want to do peer review in class, just create another .txt and name "sample_output_peer_review.txt")

3. Run the shell script
```
./hw2_seq2seq.sh [data dir] [output filename]
```
[data dir] should be "./MLDS_hw2_data" (dataset under main folder), [output filename] should be "sample_output_testset.txt"


4. (EXTRA) Peer Review: If you want to do peer review in class, run the shell script below
```
./hw2_seq2seqPR.sh [data dir] [output filename] [peer review filename]
```
[data dir] should be "./MLDS_hw2_data" (dataset under main folder), [output filename] should be "sample_output_testset.txt", and [peer review filename] is "sample_output_peer_review.txt".
## Result
