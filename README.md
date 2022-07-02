# Automatic-cell-image-classification-with-CNN

## Preprocessing
> Segmentation
>    * Using openCV and the Watershade Algorithm

<pre>
<code>
img = cv2.imread(path_dir+name)
    data_count+=1
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    ret,img_result1 = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    img_result1[985:,1262:]=0
    ##img_result2 = cv2.adaptiveThreshold(img_result1, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,7,30)
    ret, img_result2 = cv2.threshold(img_result1, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
</code>
</pre>



  ![Alt text](./img/watershed_example.jpg "segmentation_example")
