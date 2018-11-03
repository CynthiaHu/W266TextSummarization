#!/bin/bash

# clear the file
rm nyt_data.txt

# get all possible combination of year/month/day/item
# need to add error handling part if the combination is missing in the folder
for y in {1989..2007}
do
	for m in {01..12}
	do
		for d in {01..31}
		do
			# need to update the range, article in order
			for item in {0210650..0215699}
			do
				# one article per row, separate headline, lead_paragraph and full_text by tab
				unzip -p nyt_corpus_$y.zip $y/$m/$d/$item.xml | xmllint --xpath '/nitf/head/title/text()' >> nyt_data.txt
				echo -n -e "\t" >> nyt_data.txt
				unzip -p nyt_corpus_$y.zip $y/$m/$d/$item.xml | xmllint --xpath '/nitf/body/body.content/block[@class="lead_paragraph"]/p/text()' >> nyt_data.txt
				echo -n -e "\t" >> nyt_data.txt
				unzip -p nyt_corpus_$y.zip $y/$m/$d/$item.xml | xmllint --xpath '/nitf/body/body.content/block[@class="full_text"]/p/text()' >> nyt_data.txt
				echo "" >> nyt_data.txt
				
			done
		done
	done
done