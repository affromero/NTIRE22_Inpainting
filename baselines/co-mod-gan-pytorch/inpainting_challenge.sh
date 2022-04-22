root=/scratch_net/schusch_second/InpaintingChallenge
for mode in val test;
do 

    for data in Places WikiArt ImageNet;
    do
        for mask in Completion Every_N_Lines Expand MediumStrokes Nearest_Neighbor ThickStrokes ThinStrokes;
        do 
            _input_dir=$root/$data/$mode/$mask
            _output_dir=$root/$data/${mode}_CoModGAN_pretrained/$mask
            command="python test_folder.py -d ${_input_dir} -c co-mod-gan-places2-050000.pth -s ${_output_dir}"
            echo $command
            eval $command
        done
    done

    for data in FFHQ;
    do
        for mask in Completion Every_N_Lines Expand MediumStrokes Nearest_Neighbor ThickStrokes ThinStrokes;
        do 
            _input_dir=$root/$data/$mode/$mask
            _output_dir=$root/$data/${mode}_CoModGAN_pretrained/$mask
            command="python test_folder.py -d ${_input_dir} -c co-mod-gan-ffhq-9-025000.pth -s ${_output_dir}"
            echo $command
            eval $command
        done
    done

done