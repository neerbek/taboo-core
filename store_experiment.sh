#!/bin/sh

if [ "$1" = "" ]; then
    echo "store_experiment.sh <name> [<y>] [ignore_missing] [remove_intermediates]"
    echo "if second argument is set to 'y', then log files and such are goind to be deleted after this script completes"
    echo "if third argument is 'ignore_missing' then script continues even if some files are missing"
    echo "if third argument is 'remove_intermediates' then script removes all files on form save_<name>_<best|running>_*.txt"
    exit
fi

if [ "$2" = "y" ]; then
    echo "going to clean up directory"
fi

if [ "$3" != "ignore_missing" ]; then
    if [ ! -f $1.log ]; then
        echo "log file $1.log not found!"
        exit 1
    fi

    if [ ! -f save_$1_running.txt ]; then
        echo "no running model found!"
        exit 1
    fi

    if [ ! -f save_$1_best.txt ]; then
        echo "no best model found!"
        exit 1
    fi

    if [ ! -d ../taboo-jan/functionality/logs/ ]; then
        echo "log storage directory not found!"
        exit 1
    fi
fi

if [ "$3" = "remove_intermediates" ]; then
    rm save_$1_best_*.txt
    rm save_$1_running_*.txt
fi

zip $1.zip $1.log save_$1_*
mv $1.zip ../taboo-jan/functionality/logs/
svn add ../taboo-jan/functionality/logs/$1.zip
svn ci -m "added log for $1" ..
if [ "$2" = "y" ]; then
    rm $1rc
    rm $1.log
    rm save_$1_best.txt
    rm save_$1_running.txt
fi
