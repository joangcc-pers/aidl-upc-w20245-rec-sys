#!/bin/zsh

echo "Start"

echo "Downloading 2019-Oct.csv.gz"
wget https://data.rees46.com/datasets/marketplace/2019-Oct.csv.gz
gunzip 2019-Oct.csv.gz
echo "Downloading 2019-Nov.csv.gz"
wget https://data.rees46.com/datasets/marketplace/2019-Nov.csv.gz
gunzip 2019-Nov.csv.gz
echo "Downloading 2019-Dec.csv.gz"
wget https://data.rees46.com/datasets/marketplace/2019-Dec.csv.gz
gunzip 2019-Dec.csv.gz
echo "Downloading 2020-Jan.csv.gz"
wget https://data.rees46.com/datasets/marketplace/2020-Jan.csv.gz
gunzip 2020-Jan.csv.gz
echo "Downloading 2020-Feb.csv.gz"
wget https://data.rees46.com/datasets/marketplace/2020-Feb.csv.gz
gunzip 2020-Feb.csv.gz
echo "Downloading 2020-Mar.csv.gz"
wget https://data.rees46.com/datasets/marketplace/2020-Mar.csv.gz
gunzip 2020-Mar.csv.gz
echo "Downloading 2020-Apr.csv.gz"
wget https://data.rees46.com/datasets/marketplace/2020-Apr.csv.gz
gunzip 2020-Apr.csv.gz


echo "End"
