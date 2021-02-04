# pass txt_id1 as a parameter 0 to 3
N_trials=1000
counter=0
for n in range(N_trials):
    print(counter)
    counter+=1
    if counter==4:
        counter=0
