# Collect all the ace data into one directory, for convenience

ace_dir=../../../data/original/ace_2005_td_v7

original_dir=$ace_dir/data/English

out_dir=./raw_data

mkdir -p $out_dir

ls $original_dir |
    while read subdir
    do
        this_dir=$original_dir/$subdir/timex2norm
        cp $this_dir/*apf.xml $out_dir
        cp $this_dir/*.sgm $out_dir
    done
