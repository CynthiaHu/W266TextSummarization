#!/bin/bash

# The project location
cd desktop/w261submission

# The file names
fileprefix="nyt_corpus_"
filestart=1987
fileend=2007

# write to file 
rm gpfs2.map
for filename in ./gpfs2/*.zip; do
   echo $filename
   zcat $filename | awk '{print $1}' | uniq | awk '{print $1 "  gpfs2  " "'"$filename"'"}' >> gpfs2.map
done

# install xmllint in linux
# xml select is another option; below xmllint is used
# sudo apt-get install libxml2-utils
#xmllint --xpath 'string(//title)' example.xml >> test.txt

# each field separate by tab
# one line for each article
# paragraphs are not separate though!
xmllint --xpath '/nitf/head/title/text()' example.xml>> test.txt
echo -n -e "\t" >> test.txt
xmllint --xpath '/nitf/body/body.content/block[@class="lead_paragraph"]/p/text()' example.xml>> test.txt
echo -n -e "\t" >> test.txt
xmllint --xpath '/nitf/body/body.content/block[@class="full_text"]/p/text()' example.xml>> test.txt
echo "" >> test.txt
echo "test a new line" >> test.txt

# https://www.tecmint.com/linux-zcat-command-examples/

#read content of a file in a zip
unzip -p nyt_corpus_docs.zip nyt_corpus_docs/README
# same but include the file name at the beginning
unzip -c nyt_corpus_docs.zip nyt_corpus_docs/README

unzip -c nyt_corpus_1989.zip $year/$month/$day/*.xml | wc -l



for file in {example.xml, example2.xml}
do
  urlpath="$urlprefix$i$urlsuffix"
  echo $urlpath
  echo $folder
  eval "nohup wget $urlpath -P $folder/ &"
done