# Pretrained models
All model weights used in the contribution for testing are incldued in the following table:

| **Model**                  	| **Link**                   	| **ZIP size** 	|
|-------------------------------|-------------------------------|---------------|
| LightTBNet (N=4)           	| https://short.upm.es/i3xah 	| 156 MB  	    |
| DenseNet-121          	    | https://short.upm.es/z856f 	| 734 MB  	    |
| EfficientNetB0           	    | https://short.upm.es/kqirf 	| 427 MB  	    |
| EfficientNetV2-s           	| https://short.upm.es/nrrat 	| 2.11 GB  	    |
| MobileNetv3-small           	| https://short.upm.es/34l4v 	| 164 MB  	    |
| ResNet-18          	        | https://short.upm.es/vrsz1 	| 1.14 GB  	    |
| ResNet-34          	        | https://short.upm.es/n2lhl 	| 2.18 GB  	    |

---
**IMPORTANT**

**_Note 1_**: ZIP files contain weights of best and last models (5-fold CV - one model per fold).

**_Note 2_**: The default directory where to store these pretrained weights is `results/<experiment_name>`. If you store them in a different directory, beware of changing `configs -> data_in -> models_ckpts_dir` path.

Please cite us if you are using totally or partially this code and/or models.

---
## License
    
Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg




