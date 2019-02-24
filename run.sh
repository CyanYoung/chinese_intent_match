usr=`whoami`

if [[ "$usr" == "cyan" ]]
then
    python3 test.py $1 $2
else
    python test.py $1 $2
fi