# %matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import streamlit as st
import random
import json
from webcolors import rgb_to_name, hex_to_name, css3_hex_to_names, hex_to_rgb
from PIL import ImageColor
from scipy.spatial import KDTree

 


IMAGE_SIZE = (10,10)
plt.figure(figsize=IMAGE_SIZE)

class Inference():
    
    
    def __init__(self):
        '''
        
        '''
        

    def convert_rgb_to_names(self, rgb_tuple):
    
    # a dictionary of all the hex and their respective names in css3
        css3_db = css3_hex_to_names
        names = []
        rgb_values = []
        for color_hex, color_name in css3_db.items():
            names.append(color_name)
            rgb_values.append(hex_to_rgb(color_hex))
        
        kdt_db = KDTree(rgb_values)
        distance, index = kdt_db.query(rgb_tuple)
        return names[index]

    def total_value(self, n):
        return n



    def load_image(self, sample_image, resp, n, categ):
        
        img_np=mpimg.imread(sample_image)
        img = Image.fromarray(img_np.astype('uint8'),'RGB')
        x, y = img.size

        fig,ax = plt.subplots(1, figsize=(10,10))
        # Display the image
        ax.imshow(img_np)

        # draw box and label for each detection
        
        detections = json.loads(resp)
        
        for detect in detections['boxes']:
            color = ["#"+''.join([random.choice('ABCDEF0123456789') for i in range(6)])]
            color_2= ImageColor.getcolor(color[0], "RGB")
            color_name = self.convert_rgb_to_names(color_2)
            # self.color_name = color_name
            label = detect['label']
            box = detect['box']
            conf_score = detect['score']
            detect['color'] = color_name

            
            if conf_score > 0.5:
                ymin, xmin, ymax, xmax =  box['topY'],box['topX'], box['bottomY'],box['bottomX']
                topleft_x, topleft_y = x * xmin, y * ymin
                width, height = x * (xmax - xmin), y * (ymax - ymin)
                # color = np.random.rand(3) #'red'
                

                print('{}: [{}, {}, {}, {}], {}'.format(detect['label'], round(topleft_x, 3),
                                                        round(topleft_y, 3), round(width, 3),
                                                        round(height, 3), round(conf_score, 3)))

                
                rect = patches.Rectangle((topleft_x, topleft_y), width, height,
                                        linewidth=3, edgecolor=color[0],facecolor='none')
                ax.add_patch(rect)
                plt.text(topleft_x, topleft_y - 10, label, color=color[0], fontsize=20)
        
        resp_format = json.dumps(detections, indent=4, sort_keys=True)
        format_box = json.loads(resp_format)

        for det in format_box["boxes"]:
            label = det["label"]
            if label == categ:
                n +=1
                gtruth = categ
                predicted = label
            
            else:
                gtruth = categ
                predicted = label

            
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
        st.write(json.loads(resp_format))
        return self.total_value(n), gtruth, predicted
        
#fig.savefig('new.jpeg')    