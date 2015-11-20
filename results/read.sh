filename='Z_path_mine-24-4'

echo > temp_file
for ((i=4; i<245;i+=5))
do
        sed -n "$i p" $filename >> temp_file
done
sed -i -e 's/Time:/ /g' temp_file
