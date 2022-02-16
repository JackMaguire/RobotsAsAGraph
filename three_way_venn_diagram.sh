#!/bin/bash

datafile="./what_did_dfo_learn.uniq.concise.csv"

#tele level my_move pred1_move pred2_move

total_n_samples=$(grep -v tele $datafile | wc -l)

all_agree=$(grep -v tele $datafile | awk '$5 == $4 && $4 == $3' | wc -l)
echo "all_agree: $all_agree"
echo "total_n_samples: $total_n_samples"

echo "Title: Total_Agree Total_Disagree Exclusive_Agree"

pred1_agrees_with_me=$(grep -v tele $datafile | awk '$3 == $4' | wc -l)
pred1_disagrees_with_me=$(echo $total_n_samples - $pred1_agrees_with_me | bc -l)
pred1_agrees_with_me_and_not_pred2=$(echo $pred1_agrees_with_me - $all_agree | bc -l)
echo "pred1&me: $pred1_agrees_with_me $pred1_disagrees_with_me $pred1_agrees_with_me_and_not_pred2"

pred2_agrees_with_me=$(grep -v tele $datafile | awk '$3 == $5' | wc -l)
pred2_disagrees_with_me=$(echo $total_n_samples - $pred2_agrees_with_me | bc -l)
pred2_agrees_with_me_and_not_pred1=$(echo $pred2_agrees_with_me - $all_agree | bc -l)
echo "pred2&me: $pred2_agrees_with_me $pred2_disagrees_with_me $pred2_agrees_with_me_and_not_pred1"

pred1_agrees_with_pred2=$(grep -v tele $datafile | awk '$5 == $4' | wc -l)
pred1_disagrees_with_pred2=$(echo $total_n_samples - $pred1_agrees_with_pred2 | bc -l)
pred1_agrees_with_pred2_and_not_me=$(echo $pred1_agrees_with_pred2 - $all_agree | bc -l)
echo "pred1&pred2: $pred1_agrees_with_pred2 $pred1_disagrees_with_pred2 $pred1_agrees_with_pred2_and_not_me"

echo "Sole Decisions:"
all_disagree_with_me=$(grep -v tele $datafile | awk '$3 != $4 && $3 != $5' | wc -l)
all_disagree_with_pred1=$(grep -v tele $datafile | awk '$3 != $4 && $4 != $5' | wc -l)
all_disagree_with_pred2=$(grep -v tele $datafile | awk '$5 != $4 && $3 != $5' | wc -l)

echo $all_disagree_with_me $all_disagree_with_pred1 $all_disagree_with_pred2
