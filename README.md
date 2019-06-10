# chuangzhi_dawnbench
# Chuangzhi and MLU 100
Chuangzhi edge server is a new high-performance artificial intelligence (AI) platform for edge computing applications, the application domain of Chuangzhi edge server mainly focus on robotics, self-driving cars, and smart cities. A dedicated Machine Learning Unit (MLU) -- Cambricon 100 is deployed on the server to accelerate the speed of inference progress, Cambricon 100 is designed by Cambricon Technologies Corp, it is a cloud-based AI chip designed for data center workloads.

# Run inference
## unzip cambricon model
```
unzip mlu_int8_resnet50_batch1_mp16_CNML_CPU_MLU_BALANCE_fusion_1.cambricon.zip 
```

## run inference
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

# team members
{yinhes, wangying2009, quanzhenyu, lijiajun01, zhanghaotian}@ict.ac.cn
