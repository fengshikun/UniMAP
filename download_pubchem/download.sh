#!/bin/bash
# rlaunch --private-machine=group --charged-group=health --cpu=16 --gpu=0 --memory=100000 -- zsh download.sh
# there must be enough memory to support multiprocessing.
# 0413: 5 cols: Preferred Canonical Formula Mass Log P
MIN=5
MAX=5 # 1555

PREFIX="ftp://ftp.ncbi.nlm.nih.gov/pubchem/Compound/CURRENT-Full/XML/"
# fill this in
DOWNLOAD_DIR="/sharefs/chem_data/pubchem/"
EXTRACTION_FILE='names_properties_8_cols.txt' #'iupacs_properties_10.txt'
FINAL_OUTPUT_DIR='data_1m_8cols/'
mkdir -p ${DOWNLOAD_DIR}${FINAL_OUTPUT_DIR}

prev_num="0000"
for i in $(seq $MIN 5 $MAX); do

    num=$(printf "%04d" $i)
    echo $num
    fn="Compound_${prev_num}00001_${num}00000.xml"
    prev_num=$num
    echo "getting" $fn
    if ! [[ -f $DOWNLOAD_DIR$fn ]]; then
        orig_dir=$(pwd)  
        cd $DOWNLOAD_DIR
        wget "${PREFIX}${fn}.gz"
        wget "${PREFIX}${fn}.gz.md5"
        if md5sum -c ${fn}.gz.md5; then
            echo md5 passed
            # rm ${fn}.gz.md5
            # pigz does multithreaded unzipping. If you don't have pigz,
            # you can use gunzip by uncommenting the line below
            gunzip $fn
            # pigz -d -p 8 $fn
        else
            echo md5 failed
        fi
        cd $orig_dir
    fi
    echo "extracting"
    # python extract_info.py $DOWNLOAD_DIR$fn "<PC-Compound>" Preferred 11 34 -26 Traditional 11 34 -26 "Canonical<" 11 34 -26 Mass 12 34 -26 Formula 11 34 -26 "Log P" 11 34 -26 >> ${DOWNLOAD_DIR}iupacs_properties.txt
    # rm $DOWNLOAD_DIR$fn
    # echo ${DOWNLOAD_DIR}${EXTRACTION_FILE}
    echo ${DOWNLOAD_DIR}${FINAL_OUTPUT_DIR}${EXTRACTION_FILE}
    echo $DOWNLOAD_DIR$fn
    #python extract_info.py $DOWNLOAD_DIR$fn "<PC-Compound>" Preferred 11 34 -26 "Canonical<" 11 34 -26 Formula 11 34 -26 >> ${DOWNLOAD_DIR}${EXTRACTION_FILE} #iupacs_properties.txt
    python extract_info.py $DOWNLOAD_DIR$fn "<PC-Compound>" Preferred 11 34 -26 'CAS-like Style' 11 34 -26 Systematic 11 34 -26 Traditional 11 34 -26 "Canonical<" 11 34 -26 Formula 11 34 -26 Mass 12 34 -26 "Log P" 11 34 -26 >> ${DOWNLOAD_DIR}${FINAL_OUTPUT_DIR}${EXTRACTION_FILE}
    
    wc -l ${DOWNLOAD_DIR}${FINAL_OUTPUT_DIR}${EXTRACTION_FILE}
    
done

python txt2csv.py \
--input_dir=${DOWNLOAD_DIR}${FINAL_OUTPUT_DIR}${EXTRACTION_FILE} \
--output_dir=${DOWNLOAD_DIR}${FINAL_OUTPUT_DIR}
