## Code Structure

![](https://github.com/SherlockHolmes221/pytorch-faster-rcnn/raw/0.4/data/imgs/fasterrcnn_code_net.png)

- train test

![](https://github.com/SherlockHolmes221/pytorch-faster-rcnn/raw/0.4/data/imgs/fasterrcnn_code_train.png)

- loss function

![](https://github.com/SherlockHolmes221/pytorch-faster-rcnn/raw/0.4/data/imgs/fasterrcnn_code_loss.png)

## Repetition  
```buildoutcfg
# set up environment
conda create -n fasterrcnn python=3.6.9
conda activate fasterrcnn

conda install pytorch=0.4.0 torchvision cuda90 -c pytorch
pip install opencv-python
pip install easydict
pip install tensorboard-pytorch
pip install torchvision
pip install scipy
pip install pycocotools
pip install matplotlib
pip install pillow
pip installtrensorflow
pip install cython
```
```buildoutcfg
# build for nms and roi_pooling 
cd lib
bash make.sh
cd ..
```
```buildoutcfg
# build for cocoAPI
cd data
git clone https://github.com/pdollar/coco.git
cd coco/PythonAPI
make
cd ../../..
```
```buildoutcfg
# VOC data (download it in your own dir)
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_11-May-2012.tar

tar xvf VOCtrainval_06-Nov-2007.tar
tar xvf VOCtest_06-Nov-2007.tar
tar xvf VOCdevkit_08-Jun-2007.tar
tar xvf VOCtrainval_11-May-2012.tar

# As for the coco2014 dataset, Please find the links and download the by your own.
```
```buildoutcfg
# set a soft link to your data
cd data
ln -s your/voc2007and2012/path/ VOCdevkit
ln -s your/coco2014/data/path coco
```
```buildoutcfg
# downliad pretrained model
mkdir -p data/imagenet_weights
cd data/imagenet_weights
# https://pan.baidu.com/s/10psfjMoENBgH_a4rGbbejQ#list/path=%2F
```
```buildoutcfg
# The download vgg16 needs to be modified
python 
>>> import torch
>>> model = torch.load("vgg16.pth")
>>> model['classifier.0.weight'] = model['classifier.1.weight']
>>> model['classifier.0.bias'] = model['classifier.1.bias']
>>> del model['classifier.1.weight']
>>> del model['classifier.1.bias']

>>> model['classifier.3.weight'] = model['classifier.4.weight']
>>> model['classifier.3.bias'] = model['classifier.4.bias']
>>> del model['classifier.4.weight']
>>> del model['classifier.4.bias']
>>> torch.save(model, 'vgg16.pth')
```
```buildoutcfg
# train for voc
python ./tools/trainval_net.py --weight data/imagenet_weights/resnet101.pth --imdb voc_2007_trainval --imdbval voc_2007_test --iters 70000 --cfg experiments/cfgs/res101.yml  --tag  experiments/logs/res101_voc07/ --net res101 --set ANCHOR_SCALES [8,16,32] ANCHOR_RATIOS [0.5,1,2] TRAIN.STEPSIZE [50000]
python ./tools/trainval_net.py --weight data/imagenet_weights/resnet101.pth --imdb voc_2007_trainval+voc_2012_trainval --imdbval voc_2007_test --iters 110000 --cfg experiments/cfgs/res101.yml  --tag  experiments/logs/res101_voc0712/ --net res101 --set ANCHOR_SCALES [8,16,32] ANCHOR_RATIOS [0.5,1,2] TRAIN.STEPSIZE [80000]

# train for coco 
python ./tools/trainval_net.py --weight data/imagenet_weights/resnet101.pth --imdb coco_2014_train+coco_2014_valminusminival --imdbval coco_2014_minival --iters 490000 --cfg experiments/cfgs/res101.yml  --tag  experiments/logs/res101_2014coco/ --net res101 --set ANCHOR_SCALES [4,8,16,32] ANCHOR_RATIOS [0.5,1,2] TRAIN.STEPSIZE [350000]
python ./tools/trainval_net.py --weight data/imagenet_weights/resnet101.pth --imdb coco_2014_train+coco_2014_valminusminival --imdbval coco_2014_minival --iters 1190000 --cfg experiments/cfgs/res101.yml  --tag  experiments/logs/res101_2014coco/ --net res101 --set ANCHOR_SCALES [4,8,16,32] ANCHOR_RATIOS [0.5,1,2] TRAIN.STEPSIZE [900000]
```
```buildoutcfg
# evaluation(need to change according to your own path)
python ./tools/test_net.py --imdb voc_2007_test --model output/mobile/voc_2007_trainval/experiments/logs/mobile/mobile_faster_rcnn_iter_70000.pth  --cfg experiments/cfgs/mobile.yml  --tag  experiments/logs/mobile/ --net mobile --set ANCHOR_SCALES [4,8,16,32] ANCHOR_RATIOS [0.5,1,2] 
```
```buildoutcfg
# reval model
python ./tools/reval.py output/mobile/voc_2007_test/experiments/logs/mobile/mobile_faster_rcnn_iter_70000/  --imdb voc_2007_test
```
```buildoutcfg
# visualization
tensorboard --logdir=tensorboard/mobile/voc_2007_trainval/ --port=7001
```
## Project Structure and Explanations
```buildoutcfg
- data      the softlink of your train data

- docker    Dockerfile for build 

- exprerimwnts/cfgs   save some configurations for different nets

- tools     the entrance of train and test a model
  - __init_paths.py  add the project path and cocoAPI path as system paths
  - test_net.py 
     need arguments: --cfg --model --imdb --comp --num_dets --tag --net --set
     get a netword  net.create_carchotecture and load_state_dict
     jump to test_net
  - trainval_net.py 
     need_arguments: --cfg --weight --imdb --imdbval --iters -- tag --net --set
     combined_roidb   Combine multiple roidbs
     get_training_roidb  flip the image 
     prepare_roidb  format the roidb[i] with{
                                             'image':image_path
                                             'width'
                                             'height'
                                             'gt_overlaps'
                                             'max_classes' :gt_overlaps.argmax(axis=1)
                                             'max_overlaps':gt_overlaps.max(axis=1)
                                            }
     filter_roidb  BG_THRESH_LO < max_overlaps < BG_THRESH_HI or max_overlaps > FG_THRESH
```
![](https://github.com/SherlockHolmes221/CNNs/raw/master/img/code_rpn.png)
``` buildoutcfg
- lib/nets
  - network.py
     How the data flow:
     train or test:
     1. train_step_with_summary  train_step  
        self.forward, get_loss, train_op.zero_grad(),backward,train_op.step(),_run_summary_op and delete_intermediate_states
     
     2. self.forward
        get_data 'image' 'gt_boxes' 'im_info'
        _predict
        'Train':  self._add_losses() 
        'Test':   self._predictions["bbox_pred"] = bbox_pred.mul(stds).add(means)
     
     3. _predict 
        (1) self._image_to_head()
            prepare_anchors: self._anchor_component(net_conv.size(2), net_conv.size(3))
        (2)rpn: rois = self._region_proposal(net_conv)
        (3)pool5 = self._crop_pool_layer(net_conv, rois)
        (4)fc7 = self._head_to_tail(pool5)
        (5)cls_prob, bbox_pred = self._region_classification(fc7)
        
     4. _region_proposal
        (1)rpn = F.relu(self.rpn_net(net_conv))
        
        (2)rpn_cls_score = self.rpn_cls_score_net(rpn) 
        (rpn_cls_score->rpn_cls_score_reshape->rpn_cls_prob_reshape->rpn_cls_prob
        rpn_cls_score_reshape->rpn_cls_pred)
        
        (3)rpn_bbox_pred = self.rpn_bbox_pred_net(rpn) (permute(0, 2, 3, 1).contiguous())
        
        'Train':
        (4) _proposal_layer 
        (5)_anchor_target_layer  
        (6)_proposal_target_layer
        'Test':   _proposal_layer or _proposal_top_layer
        
        add self._predictions
        
     5. _crop_pool_layer
     
     6. _region_classification
       nn.Linear(self._fc7_channels, self._num_classes)
       nn.Linear(self._fc7_channels, self._num_classes * 4)
       save self._predictions["cls_score"]["cls_pred"]["cls_prob"]["bbox_pred"]
     
     7. _add_losses
       total_loss = cross_entropy + loss_box + rpn_cross_entropy + rpn_loss_box
     
     8. _anchor_component
      - generate_anchors_pre in snippets  self._anchor_length = anchors_length
      
     9. _proposal_target_layer
        proposal_target_layer in proposal_target_layer.py
     
     10. _proposal_layer
        proposal_layer in proposal_layer.py
       
     11. _anchor_target_layer
        anchor_target_layer in anchor_target_layer.py
     
     12. _proposal_top_layer
        proposal_top_layer in proposal_top_layer.py
       
     other functions:
      - create_architecture->_init_modules->init_weights(truncated normal and random normal)
      - _image_to_head and _head_to_tail need to be overwrite 
      - get_summary
      - _run_summary_op Run the summary operator: feed the placeholders with corresponding newtork outputs(activations)
        {_event_summaries, _score_summaries, _act_summaries, named_parameters}
      - _smooth_l1_loss
        in_box_diff = (bbox_pred - bbox_targets) * bbox_inside_weights
        loss_box = ( 
        (0.5 *(in_box_diff^2)*smooth_l1_sign + (abs(in_box_diff) - 0.5)*(1-smooth_l1_sign)) 
        * bbox_outside_weights
        ).sum().mean()
      
  - resnet.py  do not need maxpooling in crop_pool_layer
  - vgg16.py
      Remove fc8 Fix the layers before conv3  not using the last maxpool layer
```
``` buildoutcfg
- layer_utils
  - snippets.py 
    generate_anchors_pre  
      return len(anchor_scales) * len(anchor_ratios) * (w*16) * (h*16) anchors in different positions
  
  - proposal_layer.py  
      input:(rpn_cls_prob, rpn_bbox_pred, im_info, cfg_key, _feat_stride, anchors, num_anchors)
      select top-N according to score
      proposals = bbox_transform_inv(anchors, rpn_bbox_pred)-> gt
      select top-N after nms
  
  - proposal_top_layer.py
      proposal_top_layer(rpn_cls_prob, rpn_bbox_pred, im_imfo, anchors, num_anchors)   
      A layer that just selects the top region proposals without using non-maximal suppression
  
  - proposal_target_layer.py
      proposal_target_layer(rpn_rois, rpn_scores, gt_boxes, _num_classes)
         Assign object detection proposals to ground-truth targets. 
         add gt boxes(optional)
         # todo
         
         
      
  - anchor_target_layer.py
      anchor_target_layer(rpn_cls_score, gt_boxes, im_info, _feat_stride, all_anchors, num_anchors)
        select some anchors
        get the anchors overlaps with all gt
        get the labels[len(anchors)] = 1 positive 0 negative -1 
        bbox_targets = _compute_targets(anchors, gt_boxes[argmax_overlaps, :])
        set the inside weights and outside weights
        ( bbox_inside_weights[labels == 1, :] = np.array(cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS)
          bbox_outside_weights[labels == 1, :] = positive_weights
          bbox_outside_weights[labels == 0, :] = negative_weights
         )
         _unmap
         reshape 
```
``` buildoutcfg
- lib/roi_data_layer
  - layer.py
    RoIDataLayer
  - minibatch.py
     _get_image_blob = prep_im_for_blob + im_list_to_blob
     get_minibatch format the blobs[i]{
                                       'data'
                                       'im_info':[im_blob.shape[1], im_blob.shape[2], im_scales[0]],
                                       'gt_boxes': i * roidb[0]['boxes'] + roidb[0]['gt_classes'] (len(gt_inds), 5)
                                      }
  - roidb.py 
     prepare_roidb Enrich the imdb's roidb by adding some derived quantities 
```
``` buildoutcfg
- lib/utils
  - bbox.py 
     bbox_overlaps input: boxes: (N, 4) and query_boxes: (K, 4) return overlaps: (N, K) overlap between boxes and query_boxes
  - blob.py
     im_list_to_blob       Convert a list of images into a network input
     prep_im_for_blob      Mean subtract and scale an image for use in a blob.
  - timer.py A simple timer
  - visualization.py
     draw_bounding_boxes  draw boxes for an image and a list of boxes
     _draw_single_box     draw boxes for an image and (xmin, ymin, xmax, ymax)
```
``` buildoutcfg
- model
  - bbox_transform.py
      bbox_transform(ex_rois, gt_rois)
         To get the target box according to the paper's formula
      bbox_transform_inv(anchors, rpn_bbox_pred)
         To get the pred_boxes according to formula
      clip_boxes(boxes, im_shape)
        Clip boxes to image boundaries.
```

