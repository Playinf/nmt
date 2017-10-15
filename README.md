# RNNsearch
An implementation of RNNsearch using Theano, the implementation is identical
to [GroundHog](https://github.com/lisa-groundhog/GroundHog).

## Note
This repository is deprecated. See [RNNsearch](https://github.com/XMUNLP/RNNsearch)

## Usage

### Data Preprocessing
1. Build vocabulary
  * Build source vocabulary
  ```
  python scripts/buildvocab.py --corpus zh.txt --output vocab.zh.pkl
                               --limit 30000 --groundhog
  ```
  * Build target vocabulary
  ```
  python scripts/buildvocab.py --corpus en.txt --output vocab.en.pkl
                               --limit 30000 --groundhog
  ```
2. Shuffle corpus (Optional)
```
python scripts/shuffle.py --corpus zh.txt en.txt
```

### Training
* Training from random initialization
```
  python rnnsearch.py train --corpus zh.txt.shuf en.txt.shuf
    --vocab zh.vocab.pkl en.vocab.pkl --model nmt --embdim 620 620
    --hidden 1000 1000 1000 --maxhid 500 --deephid 620 --maxpart 2
    --alpha 5e-4 --norm 1.0 --batch 128 --maxepoch 5 --seed 1234
    --freq 1000 --vfreq 1500 --sfreq 50 --sort 20 --validation nist02.src
    --references nist02.ref0 nist02.ref1 nist02.ref2 nist02.ref3
```
* Initialize with a trained model
```
  python rnnsearch.py train --corpus zh.txt.shuf en.txt.shuf
    --vocab zh.vocab.pkl en.vocab.pkl --model nmt --embdim 620 620
    --hidden 1000 1000 1000 --maxhid 500 --deephid 620 --maxpart 2
    --alpha 5e-4 --norm 1.0 --batch 128 --maxepoch 5 --seed 1234
    --freq 1000 --vfreq 1500 --sfreq 50 --sort 20 --validation nist02.src
    --references nist02.ref0 nist02.ref1 nist02.ref2 nist02.ref3
    --initialize nmt.best.pkl
```
* Resume training
```
  python rnnsearch.py train --model nmt.autosave.pkl
```

### Decoding
```
  python rnnsearch.py translate --model nmt.best.pkl < input > translation
```

### Convert Trained Models
Models trained by GroundHog can be converted to our format using convert.py,
only support RNNsearch architecture
```
python scripts/convert.py --state search_state.pkl --model search_model.npz
                          --output nmt.pkl
```

### Convert Old Models
Convert models trained by old versions
```
python scripts/convert_model.py oldmodel.pkl newmodel.pkl
```
