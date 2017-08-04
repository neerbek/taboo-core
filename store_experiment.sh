#!/bin/sh

if [ "$1" = "" ]; then
    echo "store_experiment.sh <name> [<y>]"
    echo "if second argument is set, then log files and such are goind to be deleted after this script completes"
  exit
fi

if [ "$2" != "" ]; then
  echo "going to clean up directory"
fi

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


zip $1.zip $1.log save_$1_*
mv $1.zip ../taboo-jan/functionality/logs/
svn add ../taboo-jan/functionality/logs/$1.zip
svn ci -m "added log for $1" ..
if [ "$2" != "" ]; then
    rm $1rc
    rm $1.log
    rm save_$1_best.txt
    rm save_$1_running.txt
fi
