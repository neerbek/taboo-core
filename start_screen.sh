#!/bin/sh
if [ "$1" = "" ]; then
  echo "create_screen.sh <name>"
  exit
fi
echo logfile $1.log > $1rc
screen -S $1 -c $1rc -L
  
