#!/bin/bash

DAVX512=8
UNROLLY=8
UNROLLX=3
samples=0
for ((i=24;i<=40;i+=2));
do
    for ((j=8;j<=15;j++));
    do
        for((k=32;k<=96;k+=16));
        do
            I=`expr $i \* $UNROLLY`
            J=`expr $j \* $DAVX512`
            J=`expr $J \* $UNROLLX`
            K=$k
            all=`expr $I \* $J`
            all=`expr $all \* $K`
            # echo $all
            if [[ `expr 1024 \* 1024` -lt $all && $all -lt `expr 10240 \* 1024` ]];then
                name="$I"-"$J"-"$K".txt
                echo $name;
                samples=`expr $samples + 1`
                ./benchmark-blocked-1 $I $J $K > ./log/$name
            fi
        done
    done
done

echo "samples="$samples