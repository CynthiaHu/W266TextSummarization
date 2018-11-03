#!/bin/bash

# for folder in ./*.zip; do
   # echo $folder
   # if [ $folder -ge 1989 && $foler -le 2007 ]; then
   # echo "Couldnt find more words, exiting.."
   # exit 1

   # zcat $filename | awk '{print $1}' | uniq | awk '{print $1 "  gpfs2  " "'"$filename"'"}' >> gpfs2.map
   # fi
# done

# for y in {1989..2007}
# do
	# for m in {01..12}
	# do
		# for d in {01..31}
		# do
			# echo "$y/$m/$d"
		# done
	# done
	
# done

y=1989
wc -l nyt_corpus_$y.zip