# sumaAI

Download data from here
https://drive.google.com/drive/folders/1BUvnfLVzl34CTk3eE5lUc69PXvSx_4LT?usp=sharing

①make splited a lot of pictures from one image

```
python utils.py
```

To use torchvision.datasets.ImageFolder, utils.py make label folder.
After making folders, create a lot of pictures from one image

②before you train, make directory to save weight

```
mkdir weight
python train.py
```

③predict test picture

```
python predict.py test.jpg
```

you need to make "picture" directory and put test image there.
Then choose the image to predict by CNN model
