#sudo ln -s ~/Desktop/tensorflow/bazel-bin/tensorflow/examples/label_image /usr/local/bin/label_image
cd ~/Desktop/raw_images/raw_images_grey/images_test/brng_test
for i in $(ls); do
label_image --graph=/home/zhenyu/Desktop/Tasks/image_recognition/GoogleNet/trained_models/tmp_colored_bgd/output_graph.pb --labels=/home/zhenyu/Desktop/Tasks/image_recognition/GoogleNet/trained_models/tmp_colored_bgd/output_labels.txt --output_layer=final_result --image=$i
echo '***********************************************************************';
done
