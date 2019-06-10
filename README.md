# Chuangzhi and MLU 100
ChuangZhi edge server concentrates on providing high-performance artificial intelligence platform for edge computing. This server equips a Deep Learning accelerators called Cambricon 100, which is designed by Cambricon Technologies Co. Ltd. Cambricon 100 focus on solving demands of Cloud platform. With high throughput, Cambricon 100 has excellent performance when dealing with high-load operations. When deploying Cambricon 100 into edge computing, Cambricon 100 leaves a few to be improved in inference latency. Overall, as a powerful Deep Learning accelerator, Cambricon 100 satisfies most of the application scenarios.

# Run inference
## unzip cambricon model
```
unzip mlu_int8_resnet50_batch1_mp16_CNML_CPU_MLU_BALANCE_fusion_1.cambricon.zip 
```

## run classification
```
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
```

# Team members
{yinhes, wangying2009, quanzhenyu, lijiajun01, zhanghaotian}@ict.ac.cn
