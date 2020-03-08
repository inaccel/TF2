#./bin/host ../host/model/googlenet/param.bin ../host/model/googlenet_Q ../host/test_images/googlenet_data_label_391.bin ../host/model/googlenet/loss3_classifier.bin 1
#env CL_CONTEXT_EMULATOR_DEVICE_ALTERA=1 ./bin/host ../host/model/googlenet/param.bin ../host/model/googlenet_Q ../host/test_images/googlenet_data_label_391.bin ../host/verify/googlenet_fc1000_label_391.bin 3
#./bin/host ../host/model/googlenet/param.bin ../host/model/googlenet_Q ../host/test_images/googlenet_data_label_391.bin ../host/verify/googlenet_fc1000_label_391.bin 16
#env CL_CONTEXT_EMULATOR_DEVICE_ALTERA=1 ./bin/host ../host/model/resnet50/param.bin ../host/model/resnet50_Q ../host/test_images/resnet50_data_label_100.bin ../host/verify/resnet50_fc1000_label_100.bin 2
#./bin/host ../host/model/resnet50/param.bin ../host/model/resnet50_Q ../host/test_images/resnet50_data_label_100.bin ../host/verify/resnet50_fc1000_label_100.bin 1
#env CL_CONTEXT_EMULATOR_DEVICE_ALTERA=1 ./bin/host ../host/model/resnet50_pruned/param.bin ../host/model/resnet50_pruned_Q ../host/model/resnet50_pruned/data.bin ../host/model/resnet50_pruned/fc1000.bin 2
nohup env CL_CONTEXT_EMULATOR_DEVICE_ALTERA=1 ./bin/host ../host/model/pytorch_resnet50/param.bin ../host/model/pytorch_resnet50_q ../host/model/pytorch_resnet50/data.bin ../host/model/pytorch_resnet50/pool1.bin 1 &
#./bin/host ../host/model/resnet50_pruned/param.bin ../host/model/resnet50_pruned_Q ../host/model/resnet50_pruned/data.bin ../host/model/resnet50_pruned/fc1000.bin 16
