## CNN Model for MWP
### Requirement
pytorch v0.2.0+5c43fd (Don't use the version on official website)  
pandas  
numpy  
sklearn  
ipython  
```
# sudo apt install cmake
git clone 'https://github.com/pytorch/pytorch.git'
git checkout 5c43fcd
python setup.py install --user
```
### Interactive Usage
```
predict('Body', 'Question)
```

### Usage
```
usage: cnn.py [-h] [--test TEST] [--valid_ratio VALID_RATIO]
              [--n_iters N_ITERS] [--lr LR] [--reg REG]
              [--batch_size BATCH_SIZE]
              train embedding
You can include multiple testset for evaluation by specify --test mulitple times
```


