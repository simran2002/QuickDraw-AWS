---
license: apache-2.0
tags:
- generated_from_trainer
metrics:
- accuracy
model-index:
- name: beit-sketch-classifier
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# beit-sketch-classifier

This model is a version of [microsoft/beit-base-patch16-224-pt22k-ft22k](https://huggingface.co/microsoft/beit-base-patch16-224-pt22k-ft22k) fine-tuned on a dataset of Quick!Draw! sketches (~10% of [QuickDraw's 50M sketches](https://huggingface.co/datasets/kmewhort/quickdraw-bins-50M)).
It achieves the following results on the evaluation set:
- Loss: 0.7372
- Accuracy: 0.8098

## Intended uses & limitations

It's intended to be used to classifier sketches with a line-segment input format (there's no data augmentation in the fine-tuning; the input raster images ideally need to be generated from line-vector format very similarly to the training images).

You can generate the requisite PIL images from Quickdraw `bin` format with the following:

```
# packed bytes -> dict (fro mhttps://github.com/googlecreativelab/quickdraw-dataset/blob/master/examples/binary_file_parser.py)
def unpack_drawing(file_handle):
    key_id, = unpack('Q', file_handle.read(8))
    country_code, = unpack('2s', file_handle.read(2))
    recognized, = unpack('b', file_handle.read(1))
    timestamp, = unpack('I', file_handle.read(4))
    n_strokes, = unpack('H', file_handle.read(2))
    image = []
    n_bytes = 17
    for i in range(n_strokes):
        n_points, = unpack('H', file_handle.read(2))
        fmt = str(n_points) + 'B'
        x = unpack(fmt, file_handle.read(n_points))
        y = unpack(fmt, file_handle.read(n_points))
        image.append((x, y))
        n_bytes += 2 + 2*n_points
    result = {
        'key_id': key_id,
        'country_code': country_code,
        'recognized': recognized,
        'timestamp': timestamp,
        'image': image,
    }
    return result

# packed bin -> RGB PIL
def binToPIL(packed_drawing):
    padding = 8
    radius = 7
    scale = (224.0-(2*padding)) / 256
    
    unpacked = unpack_drawing(io.BytesIO(packed_drawing))
    unpacked_image = unpacked['image']
    image = np.full((224,224), 255, np.uint8)
    for stroke in unpacked['image']:
        prevX = round(stroke[0][0]*scale)
        prevY = round(stroke[1][0]*scale)
        for i in range(1, len(stroke[0])):
            x = round(stroke[0][i]*scale)
            y = round(stroke[1][i]*scale)
            cv2.line(image, (padding+prevX, padding+prevY), (padding+x, padding+y), 0, radius, -1)
            prevX = x
            prevY = y
    pilImage = Image.fromarray(image).convert("RGB")     
    return pilImage
```

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 64
- eval_batch_size: 64
- seed: 42
- gradient_accumulation_steps: 4
- total_train_batch_size: 256
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- lr_scheduler_warmup_ratio: 0.1
- num_epochs: 3

### Training results

| Training Loss | Epoch | Step  | Accuracy | Validation Loss |
|:-------------:|:-----:|:-----:|:--------:|:---------------:|
| 0.939         | 1.0   | 12606 | 0.7853   | 0.8275          |
| 0.7312        | 2.0   | 25212 | 0.7587   | 0.8027          |
| 0.6174        | 3.0   | 37818 | 0.7372   | 0.8098          |


### Framework versions

- Transformers 4.25.1
- Pytorch 1.13.1+cu117
- Datasets 2.7.1
- Tokenizers 0.13.2
