# mkdir data
python3 main.py --face-to-edge 0 --meshes-to-points 1 --model cgcnn --layer dynamic_edge_conv --set-x 1 --lr 0.001 --num-instances 10 --num-sample-points 512 --in-memory-dataset --batch-size 8  --use-txt
# python3 main.py --face-to-edge 0 --meshes-to-points 1 --model edge_conv --layer dynamic_edge_conv --set-x 0 --lr 0.005 --num-instances 10 --num-sample-points 512 --in-memory-dataset --batch-size 8 --nhid 256 --load-latest --mode submit
# python3 main.py --face-to-edge 0 --meshes-to-points 1 --model edge_conv --layer dynamic_edge_conv --set-x 0 --lr 0.0005 --num-instances 10 --num-sample-points 512 --use-txt --batch-size 8 --save-path /content/drive/MyDrive/ --in-memory-dataset --load-latest
# python3 main.py --face-to-edge 0 --meshes-to-points 1 --model simple_edge_conv --layer dynamic_edge_conv --set-x 0 --lr 0.0005 --num-instances 10 --num-sample-points 512 --use-txt --batch-size 1 --in-memory-dataset
# python3 main.py --face-to-edge 0 --meshes-to-points 1 --model edge_conv --layer dynamic_edge_conv --set-x 0 --lr 0.0005 --num-instances 5 --num-sample-points 32 --in-memory-dataset
