from common import *

sub_path = '/home/sapark/class/ml/final/submission/val_sub.csv'
train_csv_path = '/home/sapark/class/ml/final/data/train.csv'
result_path = '/home/sapark/class/ml/final/data/valid.csv'

sub_f = open(sub_path, 'r')
train_csv_f = open(train_csv_path, 'r')

result_f = open(result_path, 'w')

sub_lines = sub_f.readlines()
train_csv_lines = train_csv_f.readlines()

cnt = 1
for i, sub_line in enumerate(sub_lines) :
    if i == 0 : continue
    if(i %100 == 0) : print('i', i+1 )
    target_session_id = sub_line.split(',')[1]
    print(target_session_id)
    state = 0
    while 1 :
        train_csv_line = train_csv_lines[cnt] 
        cur_session_id = train_csv_line.split(',')[1]
        if target_session_id  == cur_session_id :
            result_f.write(train_csv_line)
            state = 1
            cnt = cnt+1
            continue
        
        if(state == 1) :
            break
        cnt = cnt+1





