#!/bin/bash
#Run this script in the directory where all nyt zip files have been extracted.
#the files should be in year/month/day structure, having 1987 to 2007 as the year.

OUTPUT_FILE="nyt_structured_data.csv"
# clear the file
if [ -e "$OUTPUT_FILE" ] ; then 
	mv "$OUTPUT_FILE" "$OUTPUT_FILE".$(stat -f "%Sc" -t "%Y%m%d%H%M" $OUTPUT_FILE)
fi

# get all possible combination of year/month/day/item
# need to add error handling part if the combination is missing in the folder
#for y in {1989..1989} #2007
find [12][90][0-9][0-9] -type f -name "*.xml" | while read xml_file
do
        TITLE=$(xmllint --xpath '/nitf/head/title/text()' $xml_file )
	if [ $? -ne 0 ]; then 
		TITLE=""
		echo $xml_file : XML_Parse Error : in TITLE
	fi
        LEAD_PARAGRAPH=$(xmllint --xpath '/nitf/body/body.content/block[@class="lead_paragraph"]/p/text()' $xml_file)
	if [ $? -ne 0 ]; then 
		LEAD_PARAGRAPH=""
		echo $xml_file : XML_Parse Error : in LEAD_PARAGRAPH
	fi
        FULL_TEXT=$(xmllint --xpath '/nitf/body/body.content/block[@class="full_text"]/p/text()' $xml_file)
	if [ $? -ne 0 ]; then 
		FULL_TEXT=""
		echo $xml_file : XML_Parse Error : in FULL_TEXT
	fi
        echo $xml_file , \""$TITLE"\" , \""$LEAD_PARAGRAPH"\" , \""$FULL_TEXT"\" | sed -e "s/\"/\'/g" >> nyt_structured_data.txt
done
