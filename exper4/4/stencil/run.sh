t_start=5
t_end=5
z_start=16
z_end=16
y_start=16
y_end=16
x_start=24
x_end=24

for((t=$t_start;t<=$t_end;t+=2));
do
    for((z=$z_start;z<=$z_end;z+=8));
    do
        for((y=$y_start;y<=$y_end;y+=8));
        do
            for((x=$x_start;x<=$x_end;x+=8));
            do
                export BT=$t
                export BZ=$z
                export BY=$y
                export BX=$x
                name="./log/$t-$x-$y-$z.txt"
                export proName="./$t-$x-$y-$z"
                echo $name
                echo $proName
                make
                sbatch -o $name benchmark.sh $proName
                sleep 25
                make clean
                sleep 5
            done
        done
    done
done