# L3D_project_SV3D_tree


# setup
1. Download dataset folder from Google Drive and move into data directory
```
mv `path_to_dataset` `path_to_project/data`
```

# split the dataset
1. split data_list.txt into train_list.txt and test_list.txt
```
python split_traintest.py ----data_list_path 'PATH_TO_DATA_LIST.txt'
```
2. Verify both txt files have been generated.


# training for unit test
1. Verify pix2pcloud for a single item from dataset.
```
python train.py --unit_test True --batch_size 1 --viz_debug True --max_iter 1000
```

![Motion from scripted interface](out/pcloud_iterations.gif)

# training full
1. Verify pix2pcloud for a single item from dataset.
```
python train.py  --batch_size 16  --max_iter 10000
```

# evaluate the model
Ensure the right checkpt.pth is set. Run python script to get F1 score.

```
python eval_model.py --type 'point' --load_checkpoint
```
![Motion from scripted interface](out/pred_train2.png)
![Motion from scripted interface](out/eval_point.png)