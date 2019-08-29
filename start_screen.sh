#!/bin/sh
if [ "$1" = "" ]; then
  echo "create_screen.sh <name>"
  exit
fi
echo logfile $1.log > $1rc
export EXP_NAME=$1
screen -S $1 -c $1rc -L
  
