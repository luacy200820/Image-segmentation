# Image-segmentation
For binary segmentation or multiclass segmentation using FCN8
  
# How to train model
`opendata_path`:  change your data path    
Default:   
```  
Number of segmentation: 4
Batch size: 4  
Color space: RGB (Choice: RGB, CIE) 
Filter: no (Choice: Reduce, bilateral filter, sharpen, median, no)
Loss function: categorical crossentropy  
EPOCHS: 150  
Height = width : 256  
```

# How to evaluate result
`name_file_n`: load model.h5    
`test_data`: load test data image   
`save_path`: save path  
