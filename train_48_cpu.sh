../../build/tools/caffe train -solver  solver_CIFAR10_48.prototxt -weights ../../models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel  2>&1 | tee log.txt
