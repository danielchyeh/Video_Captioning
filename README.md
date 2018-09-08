# Video_Captioning
MLDS2017 Project2

Project Link: https://www.csie.ntu.edu.tw/~yvchen/f106-adl/A2
## Quick start
1. Download MSVD dataset : 1450 videos for training and 100 videos for testing 
Link: https://drive.google.com/file/d/0B18IKlS3niGFNlBoaHJTY3NXUkE/view (provided by MLDS2017) and put MSVD dataset under video-captioning folder

2. Create a .txt file and name it "testing_output.txt". (Actually we have peer review section for MLDS2017, so if you want to do peer review in class, just create another .txt and name "sample_output_peer_review.txt")

3. Run the shell script
```
./s2vt_predict.sh [data dir] [output filename]
```
    
[data dir] should be "./MLDS_hw2_data" (dataset under main folder), [output filename] should be "./testing_output.txt"

Usage for training: modify mode = 0 (line 230 in s2vt_predict.py)<br><br>


4. (EXTRA) Peer Review: If you want to do peer review in class, run the shell script below
```
./s2vt_predict.sh [data dir] [output filename] [peer review filename]
```
Usage for peer review part: unblock line 19 in s2vt_predict.py (argv[3]), and modify peer_flag = 1 (line 373) 
[data dir] should be "./MLDS_hw2_data" (dataset under main folder), [output filename] should be "testing_output.txt", and [peer review filename] is "sample_output_peer_review.txt".
## Model Architecture: S2VT
S2VT seq2seq model is used in the task<br><br>
![image](https://github.com/danielchyeh/Video_Captioning/blob/master/assets/S2VT.png)
## Demo Result
![image](https://github.com/danielchyeh/Video_Captioning/blob/master/assets/dancing.gif)

Generated Caption: a man is dancing.

## References
- [paarthneekhara/text-to-image](https://github.com/wchliao/MLDS2017)

