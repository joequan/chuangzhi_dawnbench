# chuangzhi_dawnbench
''' python
./clas_offline_multicore_pipe \
-offlinemodel mlu_int8_resnet50_batch1_mp16_CNML_CPU_MLU_BALANCE_fusion_1.cambricon \
-data_parallelism 1 \
-model_parallelism 16 \
-threads 1 \
-label_file val-labels.txt \
-img_dir imagenet/ \
-images val.txt \
-scale 1.0 \
-fix8 1 \
-batch 1 \
-use_mean on \
-debug 1 \
-normalize 0 \
-mean_value 123.68,116.779,103.939 \
-stdt_value 58.393,57.12,57.375
'''
