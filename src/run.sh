#!/bin/sh
export PYTHONPATH=/usr/lib/python3.10
cd /home/ubuntu/MishnahBot/src
echo $PATH
echo $PATH > /home/ubuntu/MishnahBot/log.txt
/usr/bin/python3 -m streamlit run app.py --server.port 8080
