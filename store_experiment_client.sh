#!/bin/sh

if [ "$1" = "" ]; then
    echo "store_experiment_client.sh <name> [<y>]"
    echo "if second argument is set to 'y', then log files and such are goind to be deleted after this script completes"
    exit
fi

storagedir=/home/neerbek/jan/phd/DLP/paraphrase/taboo-jan/functionality/logs

if [ "$2" = "y" ]; then
    echo "going to clean up directory"
fi

if [ ! -f $1.log ]; then
    echo "log file $1.log not found!"
    exit 1
fi

if [ ! -f save_$1_running.txt ]; then
    echo "no running model found! save_$1_running.txt"
    exit 1
fi

if [ ! -f save_$1_best.txt ]; then
    echo "no best model found!"
    exit 1
fi

if [ ! -d $storagedir ]; then
    echo "log storage directory not found!"
    exit 1
fi

if [ -f $storagedir/$1.zip ]; then
    echo "Zip file already exists"
    exit 1
fi

zip $1.zip $1.log save_$1_*
mv $1.zip $storagedir
svn add $storagedir/$1.zip
svn ci -m "added log for $1" $storagedir
if [ "$2" = "y" ]; then
    rm $1.log
    rm save_$1_best.txt
    rm save_$1_running.txt
fi
