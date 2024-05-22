from math import sqrt
from pathlib import Path, PosixPath

import Levenshtein
import matplotlib.pyplot as plt
import numpy as np
import skimage as ski
from Bio import Align
from numpy import ndarray
from scikit_blobs import dog_blob_detection
from skimage.feature import blob_dog
from sklearn.cluster import KMeans
from sklearn.linear_model import RANSACRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from scipy import signal

# Random State to try and keep system deterministic
RANDOM_STATE = 2024

# DFN De Bruijn Sequence. Alphabet = 2, Order = 9
# Created using de_bruijn.py from DFN
DE_BRUIJN_SEQUENCE = "00000000011111111101111111001111110101111110001111101101111101001111100101111100001111011101111011001111010101111010001111001101111001001111000101111000001110111001110110101110110001110101101110101001110100101110100001110011001110010101110010001110001101110001001110000101110000001101101101001101100101101100001101011001101010101101010001101001001101000101101000001100110001100101001100100101100100001100010101100010001100001001100000101100000001010101001010100001010010001010001001010000001001001000001000100001"


class FireballPointPicker():

    def __init__(self, image: ndarray | PosixPath | str) -> None:
        if type(image) in [str, PosixPath]:
            self.image = ski.io.imread(image)
        elif type(image) is ndarray:
            self.image = image
        else:
            raise Exception("image must be file path or ndarray image")
        
        print("Making image landscape...")
        self.image = self.make_image_landscape()

        # ██████╗ ██╗      ██████╗ ██████╗     ██████╗ ███████╗████████╗███████╗ ██████╗████████╗██╗ ██████╗ ███╗   ██╗
        # ██╔══██╗██║     ██╔═══██╗██╔══██╗    ██╔══██╗██╔════╝╚══██╔══╝██╔════╝██╔════╝╚══██╔══╝██║██╔═══██╗████╗  ██║
        # ██████╔╝██║     ██║   ██║██████╔╝    ██║  ██║█████╗     ██║   █████╗  ██║        ██║   ██║██║   ██║██╔██╗ ██║
        # ██╔══██╗██║     ██║   ██║██╔══██╗    ██║  ██║██╔══╝     ██║   ██╔══╝  ██║        ██║   ██║██║   ██║██║╚██╗██║
        # ██████╔╝███████╗╚██████╔╝██████╔╝    ██████╔╝███████╗   ██║   ███████╗╚██████╗   ██║   ██║╚██████╔╝██║ ╚████║
        # ╚═════╝ ╚══════╝ ╚═════╝ ╚═════╝     ╚═════╝ ╚══════╝   ╚═╝   ╚══════╝ ╚═════╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝

        print("Retrieving nodes...")
        self.fireball_nodes = self.get_fireball_nodes()
        self.fireball_nodes = self.sort_fireball_nodes()

        # Plot the original self.image with fitted curve
        gs_kw = dict(width_ratios=[1.4, 0], height_ratios=[1, 10])
        fig, self.axd = plt.subplot_mosaic([['left', 'upper right'],
                                    ['left', 'lower right']],
                                    gridspec_kw=gs_kw, figsize=(5.5, 3.5),
                                    layout="constrained")
        self.axd['left'].imshow(self.image, cmap='gray', aspect='equal')
    
        # Plot fireball nodes in lime
        for node in self.fireball_nodes:
            x, y, r = node
            c = plt.Circle((x, y), r, color='lime', linewidth=2, fill=False)
            self.axd['left'].add_patch(c)


        fig2 = plt.figure()
        ax = fig2.add_subplot(111)
        ax.plot(range(len(self.fireball_nodes)), -self.fireball_nodes[:, 2])
        ax.set_xlabel('Fireball Node')
        ax.set_ylabel('Radius')
        ax.set_title('Fireball Node Sizes')
        ax.grid(True)

        dips = signal.find_peaks(
            -self.fireball_nodes[:, 2],
            width=(2,5),
            plateau_size=(2,5)   
        )
        print(dips)

        for dip in dips[0]:
            node = self.fireball_nodes[dip]
            x, y, r = node
            c = plt.Circle((x, y), r, color='blue', linewidth=2, fill=False)
            self.axd['left'].add_patch(c)


        # ██████╗ ███████╗███╗   ███╗ ██████╗ ██╗   ██╗███████╗                      
        # ██╔══██╗██╔════╝████╗ ████║██╔═══██╗██║   ██║██╔════╝                      
        # ██████╔╝█████╗  ██╔████╔██║██║   ██║██║   ██║█████╗                        
        # ██╔══██╗██╔══╝  ██║╚██╔╝██║██║   ██║╚██╗ ██╔╝██╔══╝                        
        # ██║  ██║███████╗██║ ╚═╝ ██║╚██████╔╝ ╚████╔╝ ███████╗                      
        # ╚═╝  ╚═╝╚══════╝╚═╝     ╚═╝ ╚═════╝   ╚═══╝  ╚══════╝                      
                                                                           
        # ███████╗███╗   ███╗ █████╗ ██╗     ██╗     
        # ██╔════╝████╗ ████║██╔══██╗██║     ██║     
        # ███████╗██╔████╔██║███████║██║     ██║     
        # ╚════██║██║╚██╔╝██║██╔══██║██║     ██║     
        # ███████║██║ ╚═╝ ██║██║  ██║███████╗███████╗
        # ╚══════╝╚═╝     ╚═╝╚═╝  ╚═╝╚══════╝╚══════╝
                                                    
        # ██████╗ ██╗      ██████╗ ██████╗ ███████╗  
        # ██╔══██╗██║     ██╔═══██╗██╔══██╗██╔════╝  
        # ██████╔╝██║     ██║   ██║██████╔╝███████╗  
        # ██╔══██╗██║     ██║   ██║██╔══██╗╚════██║  
        # ██████╔╝███████╗╚██████╔╝██████╔╝███████║  
        # ╚═════╝ ╚══════╝ ╚═════╝ ╚═════╝ ╚══════╝  
                                                                           
        self.fireball_nodes = np.delete(
            self.fireball_nodes,
            self.get_indices_unusually_small_fireballs(),
            axis=0
        )
        


        # ██████╗ ██╗███████╗████████╗ █████╗ ███╗   ██╗ ██████╗███████╗███████╗
        # ██╔══██╗██║██╔════╝╚══██╔══╝██╔══██╗████╗  ██║██╔════╝██╔════╝██╔════╝
        # ██║  ██║██║███████╗   ██║   ███████║██╔██╗ ██║██║     █████╗  ███████╗
        # ██║  ██║██║╚════██║   ██║   ██╔══██║██║╚██╗██║██║     ██╔══╝  ╚════██║
        # ██████╔╝██║███████║   ██║   ██║  ██║██║ ╚████║╚██████╗███████╗███████║
        # ╚═════╝ ╚═╝╚══════╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝╚══════╝╚══════╝
                                                                            
        # ██████╗ ███████╗████████╗██╗    ██╗███████╗███████╗███╗   ██╗         
        # ██╔══██╗██╔════╝╚══██╔══╝██║    ██║██╔════╝██╔════╝████╗  ██║         
        # ██████╔╝█████╗     ██║   ██║ █╗ ██║█████╗  █████╗  ██╔██╗ ██║         
        # ██╔══██╗██╔══╝     ██║   ██║███╗██║██╔══╝  ██╔══╝  ██║╚██╗██║         
        # ██████╔╝███████╗   ██║   ╚███╔███╔╝███████╗███████╗██║ ╚████║         
        # ╚═════╝ ╚══════╝   ╚═╝    ╚══╝╚══╝ ╚══════╝╚══════╝╚═╝  ╚═══╝         
                                                                            
        # ███╗   ██╗ ██████╗ ██████╗ ███████╗███████╗                           
        # ████╗  ██║██╔═══██╗██╔══██╗██╔════╝██╔════╝                           
        # ██╔██╗ ██║██║   ██║██║  ██║█████╗  ███████╗                           
        # ██║╚██╗██║██║   ██║██║  ██║██╔══╝  ╚════██║                           
        # ██║ ╚████║╚██████╔╝██████╔╝███████╗███████║                           
        # ╚═╝  ╚═══╝ ╚═════╝ ╚═════╝ ╚══════╝╚══════╝                           

        diffs = np.diff(self.fireball_nodes[:, :2], axis=0)
        self.distances = np.linalg.norm(diffs, axis=1)
        print("Distances:\n", self.distances, "\n")


        # ██████╗ ███████╗███╗   ███╗ ██████╗ ██╗   ██╗███████╗    ███████╗ █████╗ ██████╗ 
        # ██╔══██╗██╔════╝████╗ ████║██╔═══██╗██║   ██║██╔════╝    ██╔════╝██╔══██╗██╔══██╗
        # ██████╔╝█████╗  ██╔████╔██║██║   ██║██║   ██║█████╗      █████╗  ███████║██████╔╝
        # ██╔══██╗██╔══╝  ██║╚██╔╝██║██║   ██║╚██╗ ██╔╝██╔══╝      ██╔══╝  ██╔══██║██╔══██╗
        # ██║  ██║███████╗██║ ╚═╝ ██║╚██████╔╝ ╚████╔╝ ███████╗    ██║     ██║  ██║██║  ██║
        # ╚═╝  ╚═╝╚══════╝╚═╝     ╚═╝ ╚═════╝   ╚═══╝  ╚══════╝    ╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝
                                                                                        
        #  █████╗ ██╗    ██╗ █████╗ ██╗   ██╗    ███████╗ █████╗ ██╗     ███████╗███████╗  
        # ██╔══██╗██║    ██║██╔══██╗╚██╗ ██╔╝    ██╔════╝██╔══██╗██║     ██╔════╝██╔════╝  
        # ███████║██║ █╗ ██║███████║ ╚████╔╝     █████╗  ███████║██║     ███████╗█████╗    
        # ██╔══██║██║███╗██║██╔══██║  ╚██╔╝      ██╔══╝  ██╔══██║██║     ╚════██║██╔══╝    
        # ██║  ██║╚███╔███╔╝██║  ██║   ██║       ██║     ██║  ██║███████╗███████║███████╗  
        # ╚═╝  ╚═╝ ╚══╝╚══╝ ╚═╝  ╚═╝   ╚═╝       ╚═╝     ╚═╝  ╚═╝╚══════╝╚══════╝╚══════╝  
                                                                                        
        # ██████╗  ██████╗ ███████╗██╗████████╗██╗██╗   ██╗███████╗███████╗                
        # ██╔══██╗██╔═══██╗██╔════╝██║╚══██╔══╝██║██║   ██║██╔════╝██╔════╝                
        # ██████╔╝██║   ██║███████╗██║   ██║   ██║██║   ██║█████╗  ███████╗                
        # ██╔═══╝ ██║   ██║╚════██║██║   ██║   ██║╚██╗ ██╔╝██╔══╝  ╚════██║                
        # ██║     ╚██████╔╝███████║██║   ██║   ██║ ╚████╔╝ ███████╗███████║                
        # ╚═╝      ╚═════╝ ╚══════╝╚═╝   ╚═╝   ╚═╝  ╚═══╝  ╚══════╝╚══════╝                

        self.distances = self.remove_far_away_false_positives()

        print("Distances with outliers based on distances removed:\n", self.distances)
        print("Number of distances:", len(self.distances), "\n")


        # ██████╗ ██╗███████╗████████╗ █████╗ ███╗   ██╗ ██████╗███████╗
        # ██╔══██╗██║██╔════╝╚══██╔══╝██╔══██╗████╗  ██║██╔════╝██╔════╝
        # ██║  ██║██║███████╗   ██║   ███████║██╔██╗ ██║██║     █████╗  
        # ██║  ██║██║╚════██║   ██║   ██╔══██║██║╚██╗██║██║     ██╔══╝  
        # ██████╔╝██║███████║   ██║   ██║  ██║██║ ╚████║╚██████╗███████╗
        # ╚═════╝ ╚═╝╚══════╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝╚══════╝
                                                                    
        #  ██████╗ ██████╗  ██████╗ ██╗   ██╗██████╗ ███████╗           
        # ██╔════╝ ██╔══██╗██╔═══██╗██║   ██║██╔══██╗██╔════╝           
        # ██║  ███╗██████╔╝██║   ██║██║   ██║██████╔╝███████╗           
        # ██║   ██║██╔══██╗██║   ██║██║   ██║██╔═══╝ ╚════██║           
        # ╚██████╔╝██║  ██║╚██████╔╝╚██████╔╝██║     ███████║           
        #  ╚═════╝ ╚═╝  ╚═╝ ╚═════╝  ╚═════╝ ╚═╝     ╚══════╝           

        self.distance_groups = self.get_distance_groups()

        cumulative_node_count = 0
        for i in range(len(self.distance_groups)):
            node = self.fireball_nodes[cumulative_node_count]
            x, y, r = node
            c = plt.Circle((x, y), r, color='pink', linewidth=2, fill=False)
            self.axd['left'].add_patch(c)

            cumulative_node_count += len(self.distance_groups[i])


        # ███╗   ██╗██╗   ██╗███╗   ███╗██████╗ ███████╗██████╗     ██╗     ██╗███╗   ██╗███████╗                
        # ████╗  ██║██║   ██║████╗ ████║██╔══██╗██╔════╝██╔══██╗    ██║     ██║████╗  ██║██╔════╝                
        # ██╔██╗ ██║██║   ██║██╔████╔██║██████╔╝█████╗  ██████╔╝    ██║     ██║██╔██╗ ██║█████╗                  
        # ██║╚██╗██║██║   ██║██║╚██╔╝██║██╔══██╗██╔══╝  ██╔══██╗    ██║     ██║██║╚██╗██║██╔══╝                  
        # ██║ ╚████║╚██████╔╝██║ ╚═╝ ██║██████╔╝███████╗██║  ██║    ███████╗██║██║ ╚████║███████╗                
        # ╚═╝  ╚═══╝ ╚═════╝ ╚═╝     ╚═╝╚═════╝ ╚══════╝╚═╝  ╚═╝    ╚══════╝╚═╝╚═╝  ╚═══╝╚══════╝                
                                                                                                            
        #  █████╗ ███╗   ██╗██████╗     ██╗  ██╗██╗███████╗████████╗ ██████╗  ██████╗ ██████╗  █████╗ ███╗   ███╗
        # ██╔══██╗████╗  ██║██╔══██╗    ██║  ██║██║██╔════╝╚══██╔══╝██╔═══██╗██╔════╝ ██╔══██╗██╔══██╗████╗ ████║
        # ███████║██╔██╗ ██║██║  ██║    ███████║██║███████╗   ██║   ██║   ██║██║  ███╗██████╔╝███████║██╔████╔██║
        # ██╔══██║██║╚██╗██║██║  ██║    ██╔══██║██║╚════██║   ██║   ██║   ██║██║   ██║██╔══██╗██╔══██║██║╚██╔╝██║
        # ██║  ██║██║ ╚████║██████╔╝    ██║  ██║██║███████║   ██║   ╚██████╔╝╚██████╔╝██║  ██║██║  ██║██║ ╚═╝ ██║
        # ╚═╝  ╚═╝╚═╝  ╚═══╝╚═════╝     ╚═╝  ╚═╝╚═╝╚══════╝   ╚═╝    ╚═════╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝

        ## Plot Distances
        # Generating y values of 0 for each number
        scatter_y_values = [0] * len(self.distances)

        # Plotting the scatter plot
        self.axd['upper right'].scatter(self.distances, scatter_y_values)

        # Setting up x and y axis limits
        self.axd['upper right'].set_xlim(min(self.distances) - 1, max(self.distances) + 1)  # Adjusting x-axis limits for better visualization
        self.axd['upper right'].set_ylim(-1, 1)  # Adjusting y-axis limits to have a single line at y=0

        self.axd['upper right'].yaxis.set_ticks([])
        self.axd['upper right'].set_title("Distances Between Nodes")

        ## Plot Noramlised Frequency Histogram
        min_distance = min(self.distances)
        max_distance = max(self.distances)

        normalised = (self.distances[:] - min_distance) / (max_distance - min_distance)
        rounded = np.round(normalised[:] * 20) / 20

        print("Normalised distances:\n", rounded, "\n")

        # Plot the frequency graph
        self.axd['lower right'].hist(rounded, bins=20, align='left', edgecolor='black', density=True)
        self.axd['lower right'].set_xlabel('Number')
        self.axd['lower right'].set_ylabel('Frequency')
        self.axd['lower right'].set_title('Normalised Distances Between Nodes Frequency Histogram')
        self.axd['lower right'].grid(True)


        # ██╗███╗   ██╗██╗████████╗██╗ █████╗ ██╗                   
        # ██║████╗  ██║██║╚══██╔══╝██║██╔══██╗██║                   
        # ██║██╔██╗ ██║██║   ██║   ██║███████║██║                   
        # ██║██║╚██╗██║██║   ██║   ██║██╔══██║██║                   
        # ██║██║ ╚████║██║   ██║   ██║██║  ██║███████╗              
        # ╚═╝╚═╝  ╚═══╝╚═╝   ╚═╝   ╚═╝╚═╝  ╚═╝╚══════╝              
                                                                
        # ██╗  ██╗    ███╗   ███╗███████╗ █████╗ ███╗   ██╗███████╗ 
        # ██║ ██╔╝    ████╗ ████║██╔════╝██╔══██╗████╗  ██║██╔════╝ 
        # █████╔╝     ██╔████╔██║█████╗  ███████║██╔██╗ ██║███████╗ 
        # ██╔═██╗     ██║╚██╔╝██║██╔══╝  ██╔══██║██║╚██╗██║╚════██║ 
        # ██║  ██╗    ██║ ╚═╝ ██║███████╗██║  ██║██║ ╚████║███████║ 
        # ╚═╝  ╚═╝    ╚═╝     ╚═╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝ 
                                                                
        #  ██████╗██╗     ██╗   ██╗███████╗████████╗███████╗██████╗ 
        # ██╔════╝██║     ██║   ██║██╔════╝╚══██╔══╝██╔════╝██╔══██╗
        # ██║     ██║     ██║   ██║███████╗   ██║   █████╗  ██████╔╝
        # ██║     ██║     ██║   ██║╚════██║   ██║   ██╔══╝  ██╔══██╗
        # ╚██████╗███████╗╚██████╔╝███████║   ██║   ███████╗██║  ██║
        #  ╚═════╝╚══════╝ ╚═════╝ ╚══════╝   ╚═╝   ╚══════╝╚═╝  ╚═╝

        k_means_groups = []
        for i in range(len(self.distance_groups)):
            print(f"Group {i+1}")
            k_means_groups.append(self.k_means_distances(self.distance_groups[i]))

        self.distance_labels = []
        for _, labels in k_means_groups:
            self.distance_labels.extend(labels)
        print("Fireball distance labels after initial k means cluster:\n", self.distance_labels,"\n")


        # ██████╗ ███████╗███╗   ███╗ ██████╗ ██╗   ██╗███████╗                 
        # ██╔══██╗██╔════╝████╗ ████║██╔═══██╗██║   ██║██╔════╝                 
        # ██████╔╝█████╗  ██╔████╔██║██║   ██║██║   ██║█████╗                   
        # ██╔══██╗██╔══╝  ██║╚██╔╝██║██║   ██║╚██╗ ██╔╝██╔══╝                   
        # ██║  ██║███████╗██║ ╚═╝ ██║╚██████╔╝ ╚████╔╝ ███████╗                 
        # ╚═╝  ╚═╝╚══════╝╚═╝     ╚═╝ ╚═════╝   ╚═══╝  ╚══════╝                 
                                                                            
        # ███████╗███╗   ███╗ █████╗ ██╗     ██╗                                
        # ██╔════╝████╗ ████║██╔══██╗██║     ██║                                
        # ███████╗██╔████╔██║███████║██║     ██║                                
        # ╚════██║██║╚██╔╝██║██╔══██║██║     ██║                                
        # ███████║██║ ╚═╝ ██║██║  ██║███████╗███████╗                           
        # ╚══════╝╚═╝     ╚═╝╚═╝  ╚═╝╚══════╝╚══════╝                           
                                                                            
        # ██████╗ ██╗███████╗████████╗ █████╗ ███╗   ██╗ ██████╗███████╗███████╗
        # ██╔══██╗██║██╔════╝╚══██╔══╝██╔══██╗████╗  ██║██╔════╝██╔════╝██╔════╝
        # ██║  ██║██║███████╗   ██║   ███████║██╔██╗ ██║██║     █████╗  ███████╗
        # ██║  ██║██║╚════██║   ██║   ██╔══██║██║╚██╗██║██║     ██╔══╝  ╚════██║
        # ██████╔╝██║███████║   ██║   ██║  ██║██║ ╚████║╚██████╗███████╗███████║
        # ╚═════╝ ╚═╝╚══════╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝╚══════╝╚══════╝

        unusually_small_distance_indices = self.get_unusually_small_distances()

        fireball_indices_to_remove, distance_labels_indices_to_remove = self.get_false_positives_based_on_distance(
            unusually_small_distance_indices
        )

        print("Indicies to remove based on unusually small distance:")
        print("Fireball indices:", fireball_indices_to_remove)
        print("Distance indices:", distance_labels_indices_to_remove, "\n")

        print("No. of Distances, No. of Distance Labels, No. of Fireballs")
        print("Before deleting unusually small distances:\n", len(self.distances), len(self.distance_labels), len(self.fireball_nodes))

        self.distances = np.delete(self.distances, distance_labels_indices_to_remove)

        print("After deleting unusually small distances:\n", len(self.distances), len(self.distance_labels), len(self.fireball_nodes))

        for fireball_index in fireball_indices_to_remove:
            self.distances = np.insert(
                self.distances,
                fireball_index,
                np.sqrt(
                    np.sum(
                        (
                            self.fireball_nodes[fireball_index + 1][:2] - self.fireball_nodes[fireball_index - 1][:2]
                        )**2
                    )
                )
            )

        print("After inserting distances skipping false fireball nodes:\n", len(self.distances), len(self.distance_labels), len(self.fireball_nodes))

        self.fireball_nodes = np.delete(self.fireball_nodes, fireball_indices_to_remove, axis=0)
        print("After removing false fireball node:\n", len(self.distances), len(self.distance_labels), len(self.fireball_nodes))
        print()

        print("Final Fireballs Nodes:\n", self.fireball_nodes, "\n")


        # ██╗  ██╗    ███╗   ███╗███████╗ █████╗ ███╗   ██╗███████╗
        # ██║ ██╔╝    ████╗ ████║██╔════╝██╔══██╗████╗  ██║██╔════╝
        # █████╔╝     ██╔████╔██║█████╗  ███████║██╔██╗ ██║███████╗
        # ██╔═██╗     ██║╚██╔╝██║██╔══╝  ██╔══██║██║╚██╗██║╚════██║
        # ██║  ██╗    ██║ ╚═╝ ██║███████╗██║  ██║██║ ╚████║███████║
        # ╚═╝  ╚═╝    ╚═╝     ╚═╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝
                                                                
        #  █████╗  ██████╗  █████╗ ██╗███╗   ██╗                   
        # ██╔══██╗██╔════╝ ██╔══██╗██║████╗  ██║                   
        # ███████║██║  ███╗███████║██║██╔██╗ ██║                   
        # ██╔══██║██║   ██║██╔══██║██║██║╚██╗██║                   
        # ██║  ██║╚██████╔╝██║  ██║██║██║ ╚████║                   
        # ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝                   

        # Retrieve distance groups and distance labels again
        # now that we removed the assumed false positives
        # based on unusually small distances
        self.distance_groups = self.get_distance_groups()
        k_means_groups = []
        for i in range(len(self.distance_groups)):
            print(f"Group {i+1}")
            k_means_groups.append(self.k_means_distances(self.distance_groups[i]))

        self.distance_labels = []
        for _, labels in k_means_groups:
            self.distance_labels.extend(labels)

        print("Final Distance Labels:\n", self.distance_labels, "\n")

        print("Final No.\n", len(self.distances), len(self.distance_labels), len(self.fireball_nodes))
        print()

        for i in range(len(self.distance_labels)):
            if i + 1 == len(self.fireball_nodes):
                break
            x1, y1, _ = self.fireball_nodes[i]
            x2, y2, _ = self.fireball_nodes[i + 1]
            self.axd['left'].text(((x1 + x2) / 2), ((y1 + y2) / 2), self.distance_labels[i], color="white")

        print()
        print()


        # ███████╗███████╗ ██████╗ ██╗   ██╗███████╗███╗   ██╗ ██████╗███████╗        
        # ██╔════╝██╔════╝██╔═══██╗██║   ██║██╔════╝████╗  ██║██╔════╝██╔════╝        
        # ███████╗█████╗  ██║   ██║██║   ██║█████╗  ██╔██╗ ██║██║     █████╗          
        # ╚════██║██╔══╝  ██║▄▄ ██║██║   ██║██╔══╝  ██║╚██╗██║██║     ██╔══╝          
        # ███████║███████╗╚██████╔╝╚██████╔╝███████╗██║ ╚████║╚██████╗███████╗        
        # ╚══════╝╚══════╝ ╚══▀▀═╝  ╚═════╝ ╚══════╝╚═╝  ╚═══╝ ╚═════╝╚══════╝        
                                                                                    
        #  █████╗ ██╗     ██╗ ██████╗ ███╗   ██╗███╗   ███╗███████╗███╗   ██╗████████╗
        # ██╔══██╗██║     ██║██╔════╝ ████╗  ██║████╗ ████║██╔════╝████╗  ██║╚══██╔══╝
        # ███████║██║     ██║██║  ███╗██╔██╗ ██║██╔████╔██║█████╗  ██╔██╗ ██║   ██║   
        # ██╔══██║██║     ██║██║   ██║██║╚██╗██║██║╚██╔╝██║██╔══╝  ██║╚██╗██║   ██║   
        # ██║  ██║███████╗██║╚██████╔╝██║ ╚████║██║ ╚═╝ ██║███████╗██║ ╚████║   ██║   
        # ╚═╝  ╚═╝╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝     ╚═╝╚══════╝╚═╝  ╚═══╝   ╚═╝   

        lr_sequence = self.get_left_to_right_sequence_from_distance_labels()

        print("Left to right sequence:\n", lr_sequence)
        lr_alignment = self.perform_alignment(lr_sequence)

        rl_sequence = lr_sequence[::-1]
        print("Right to left sequence:\n", rl_sequence)
        rl_alignment = self.perform_alignment(rl_sequence)

        left_to_right = True
        fireball_sequence: str = lr_sequence
        alignment = lr_alignment

        if rl_alignment[0] > lr_alignment[0]:
            left_to_right = False
            fireball_sequence = rl_sequence
            alignment = rl_alignment

        print("Left to right?", left_to_right)
        print()

        if not left_to_right:
            self.fireball_nodes = self.fireball_nodes[::-1]
            self.distance_labels = self.distance_labels[::-1]

        self.fireball_nodes = list(self.fireball_nodes)

        print("De Bruijn Segment:", alignment[2])
        print("Fireball Sequence:", alignment[3])
        print("Distance Labels  :", "".join([str(i) for i in self.distance_labels]))
        print()


        #  █████╗ ███████╗███████╗██╗ ██████╗ ███╗   ██╗██╗███╗   ██╗ ██████╗   
        # ██╔══██╗██╔════╝██╔════╝██║██╔════╝ ████╗  ██║██║████╗  ██║██╔════╝   
        # ███████║███████╗███████╗██║██║  ███╗██╔██╗ ██║██║██╔██╗ ██║██║  ███╗  
        # ██╔══██║╚════██║╚════██║██║██║   ██║██║╚██╗██║██║██║╚██╗██║██║   ██║  
        # ██║  ██║███████║███████║██║╚██████╔╝██║ ╚████║██║██║ ╚████║╚██████╔╝  
        # ╚═╝  ╚═╝╚══════╝╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝╚═╝  ╚═══╝ ╚═════╝   
                                                                            
        # ██╗      █████╗ ██████╗ ███████╗██╗     ███████╗    ████████╗ ██████╗ 
        # ██║     ██╔══██╗██╔══██╗██╔════╝██║     ██╔════╝    ╚══██╔══╝██╔═══██╗
        # ██║     ███████║██████╔╝█████╗  ██║     ███████╗       ██║   ██║   ██║
        # ██║     ██╔══██║██╔══██╗██╔══╝  ██║     ╚════██║       ██║   ██║   ██║
        # ███████╗██║  ██║██████╔╝███████╗███████╗███████║       ██║   ╚██████╔╝
        # ╚══════╝╚═╝  ╚═╝╚═════╝ ╚══════╝╚══════╝╚══════╝       ╚═╝    ╚═════╝ 
                                                                            
        # ███╗   ██╗ ██████╗ ██████╗ ███████╗███████╗                           
        # ████╗  ██║██╔═══██╗██╔══██╗██╔════╝██╔════╝                           
        # ██╔██╗ ██║██║   ██║██║  ██║█████╗  ███████╗                           
        # ██║╚██╗██║██║   ██║██║  ██║██╔══╝  ╚════██║                           
        # ██║ ╚████║╚██████╔╝██████╔╝███████╗███████║                           
        # ╚═╝  ╚═══╝ ╚═════╝ ╚═════╝ ╚══════╝╚══════╝                           

        _, de_bruijn_start_index, de_bruijn_segment, fireball_sequence = alignment

        self.fireball_points = []
        self.fireball_nodes_queue = self.fireball_nodes.copy()
        self.fireball_labels_queue = self.distance_labels.copy()


        # Assign labels to nodes
        for de_bruijn_pos, de_bruijn_val, fireball_val in zip(
            range(
                de_bruijn_start_index,
                de_bruijn_start_index + len(de_bruijn_segment)
            ), 
            de_bruijn_segment,
            fireball_sequence
        ):

            if len(self.fireball_labels_queue) == 0:
                break

            if de_bruijn_val == "0" and fireball_val == "0":
                self.consume_node(de_bruijn_pos, de_bruijn_val)
            elif de_bruijn_val == "1" and fireball_val == "1":
                self.consume_node(de_bruijn_pos, de_bruijn_val)
                if len(self.fireball_labels_queue) == 0:
                    break

                if self.fireball_labels_queue[0] == 1:
                    self.consume_node(de_bruijn_pos, de_bruijn_val)
                else:
                    self.backtrack_odd_number_of_1s(de_bruijn_pos)

            elif de_bruijn_val == "0" and fireball_val == "1":
                self.consume_node(de_bruijn_pos, de_bruijn_val)
            elif de_bruijn_val == "1" and fireball_val == "0":
                self.consume_node(de_bruijn_pos, de_bruijn_val)
            elif de_bruijn_val == "0" and fireball_val == "-":
                pass
            elif de_bruijn_val == "1" and fireball_val == "-":
                pass


        # Assign label to the last fireball node
        last_node_de_bruijn_pos = -1

        last_point, second_last_point = self.fireball_points[-1], self.fireball_points[-2]
        if last_point[3] == '0':
            last_node_de_bruijn_pos = alignment[1] + len(alignment[2]) # next de bruijn
        else: # '1'
            if second_last_point[3] == '0':
                last_node_de_bruijn_pos = alignment[1] + len(alignment[2]) - 1 # current de bruijn
            else: # '1'
                if last_point[2] == second_last_point[2]:
                    last_node_de_bruijn_pos = alignment[1] + len(alignment[2]) # next de bruijn
                else:
                    last_node_de_bruijn_pos = alignment[1] + len(alignment[2]) - 1 # current de bruijn
        
        last_node = self.fireball_nodes_queue.pop(0)
        self.fireball_points.append(
            [
                last_node[0],
                last_node[1],
                last_node_de_bruijn_pos,
                DE_BRUIJN_SEQUENCE[last_node_de_bruijn_pos]
            ]
        )


        print("Final Fireball Points:")
        print("[x, y, de bruijn pos, 0 or 1]")
        for i in self.fireball_points:
            print(i)


    def show_plot(self):
        plt.show()


    def make_image_landscape(self):
        """
        Make sure self.image is landscape. Ensures the regression
        curve is able to properly fit the fireball and makes
        retrieving the sequence based on x coordinate clearer
        """
        height, width = self.image.shape[:2]
        self.rotated = False
        if width < height:
            # Rotate the self.image to landscape orientation
            self.image = ski.transform.rotate(self.image, angle=90, resize=True)
            self.rotated = True
        return self.image
    

    def get_fireball_nodes(self) -> ndarray:
        blobs = blob_dog(
            self.image,
            min_sigma=5,
            max_sigma=30,
            threshold=4
        )
        blobs[:, 2] = blobs[:, 2] * sqrt(2)
        print(len(blobs))

        # Extract x and y coordinates from blobs_dog
        y_coords = blobs[:, 0]
        x_coords = blobs[:, 1]

        # Fit a polynomial curve using RANSAC
        model = make_pipeline(PolynomialFeatures(2), RANSACRegressor(residual_threshold=10, max_trials=100, random_state=RANDOM_STATE))
        model.fit(x_coords.reshape(-1, 1), y_coords)

        inlier_mask = model.named_steps['ransacregressor'].inlier_mask_

        # apply inlier mask to blobs to retrieve fireball nodes
        self.fireball_nodes = np.compress(inlier_mask, blobs, axis=0)

        # Swpan columns so that x comes before y
        self.fireball_nodes[:, [0, 1]] = self.fireball_nodes[:, [1, 0]]

        return self.fireball_nodes
    

    def sort_fireball_nodes(self) -> ndarray:
        sorted_indices = np.argsort(self.fireball_nodes[:, 0])
        self.fireball_nodes = self.fireball_nodes[sorted_indices]

        print("Fireball nodes (x, y, r):\n", self.fireball_nodes, "\n")
        
        return self.fireball_nodes
    

    def get_indices_unusually_small_fireballs(self) -> list[int]:
        indices_to_remove = []
        for i in range(1, len(self.fireball_nodes) - 1):
            previous = self.fireball_nodes[i-1]
            current = self.fireball_nodes[i]
            next = self.fireball_nodes[i+1]

            average_radius_neighbours = (previous[2] + next[2]) / 2

            if current[2] < 0.40 * average_radius_neighbours:
                indices_to_remove.append(i)

        # Plot small fireballs in red
        # for i in indices_to_remove:
        #     node = self.fireball_nodes[i]
        #     x, y, r = node
        #     c = plt.Circle((x, y), r, color='red', linewidth=2, fill=False)
        #     self.axd['left'].add_patch(c)
        
        return indices_to_remove
    

    def remove_far_away_false_positives(self) -> np.ndarray:
        q25, q75 = np.percentile(self.distances, [25, 75])
        iqr = q75 - q25
        lower_bound = q25 - (1.5 * iqr)
        upper_bound = q75 + (8 * iqr)
        
        new_distances = np.array([d for d in self.distances if (lower_bound < d and d < upper_bound)])
        
        print(
            np.column_stack(
                (
                    ["q25", "q75", "iqr", "lower_bound", "upper_bound"],
                    [q25, q75, iqr, lower_bound, upper_bound]
                )
            ),
            "\n"
        )

        return new_distances
    

    def get_distance_groups(self) -> list[np.ndarray]:
        distance_groups = []

        if len(self.distances) < 40:
            return self.distances.reshape(1, -1)
        
        MIN_GROUP_NUMBER = 20

        number_of_distances = len(self.distances)
        number_of_groups = number_of_distances // MIN_GROUP_NUMBER
        remainder_from_20 = number_of_distances % MIN_GROUP_NUMBER
        number_extra_from_remainder = remainder_from_20 // number_of_groups

        base_group_size = MIN_GROUP_NUMBER + number_extra_from_remainder
        print("Base group size:", base_group_size)

        for i in range(number_of_groups):
            if i < number_of_groups - 1:
                group = self.distances[i * base_group_size: (i + 1) * base_group_size]
            else:
                group = self.distances[i * base_group_size: number_of_distances]
            distance_groups.append(group)

        print("Distance groups:\n", distance_groups)

        return distance_groups
    

    def k_means_distances(self, input_distances: ndarray) -> tuple[list[float], list[int]]:
        reshaped_distances = np.array(input_distances).reshape(-1, 1)
        kmeans = KMeans(2, random_state=RANDOM_STATE)
        kmeans.fit(reshaped_distances)

        # Get cluster centers
        cluster_centers = kmeans.cluster_centers_[:,0]
        # Get cluster labels for each data point
        labels = kmeans.labels_

        # make it so that the 'longer' distances get labelled with a 1
        # and 'shorter' distances get labelled with a 2

        node_distances_labels = labels
        # cluster labels are either 0 or 1 depending on which label goes first.
        # so check if the shorter labels came first and to do a bitwise XOR on them.
        if cluster_centers[0] < cluster_centers[1]:
            node_distances_labels ^= 1
        else:
            cluster_centers[0], cluster_centers[1] = cluster_centers[1], cluster_centers[0]

        print("Cluster centers:\n", cluster_centers)
        # 1 = long, 0 = short
        print("Distance labels:\n", node_distances_labels, "\n")

        return (cluster_centers, node_distances_labels)
    

    def get_unusually_small_distances(self):
        """
        Based on the small gaps found using k means cluster,
        check every gap to see if it's unusually small.
        Assume an unusually small gap means that one of
        the nodes that created this distance is a false positive.

        NOTE: What if there are many false positives in a row?
        Currently not taken into account.
        """

        # threshold used to check if current distance is
        # unusually small compared to the average of its
        # neighbours
        SIZE_THRESHOLD = 0.7

        small_gaps_mask_array = np.array(self.distance_labels, dtype=bool)
        small_gaps = self.distances[small_gaps_mask_array]
        
        print("Small gaps to check:\n", small_gaps, "\n")

        small_gaps_length = len(small_gaps)

        original_indices = [i for i, m in enumerate(self.distance_labels) if m]

        # consider 6 neighbouring distances to check if the current
        # distance is unusually small
        window_size = 7

        unusually_small_distance_indices = []

        for i, gap in enumerate(small_gaps):
            left_idx = i - (window_size // 2)
            right_idx = i + (window_size // 2) + 1

            # ensure a window of 7 is retrieved even if on the ends
            if left_idx < 0:
                right_idx += abs(left_idx)
                left_idx = 0
            
            if right_idx > small_gaps_length + 1:
                left_idx -= (right_idx - small_gaps_length)
                right_idx = small_gaps_length
            
            # retrieve window of gaps, get mean
            window = small_gaps[left_idx:right_idx]
            window_mean = np.mean(window)

            if gap < window_mean * SIZE_THRESHOLD:
                unusually_small_distance_indices.append(original_indices[i])

        return unusually_small_distance_indices
    

    def get_false_positives_based_on_distance(self, small_distance_indices: list) -> tuple[list[int], list[int]]:
        """
        Retrieves the indices of the false positive fireballs
        and their corresponding distances.
        """

        fireball_indices_to_remove = []
        new_distance_labels_indices_to_remove = []

        for ii in small_distance_indices:

            if ii not in new_distance_labels_indices_to_remove:
                new_distance_labels_indices_to_remove.append(ii)
            
            fireball_index_to_remove = -1
            
            # a distance value has a left node and right node.
            if self.fireball_nodes[ii][2] < self.fireball_nodes[ii + 1][2]:
                # left node smaller than right node.
                # treat left node as false positive
                fireball_index_to_remove = ii
                if (ii - 1) not in new_distance_labels_indices_to_remove:
                    new_distance_labels_indices_to_remove.append(ii - 1)
            else:
                # right node smaller than left node
                # treat right node as false positive
                fireball_index_to_remove = ii + 1
                if (ii + 1) not in new_distance_labels_indices_to_remove:
                    new_distance_labels_indices_to_remove.append(ii + 1)

            x, y, r = self.fireball_nodes[fireball_index_to_remove]
            c = plt.Circle((x, y), r, color='orange', linewidth=2, fill=False)
            self.axd['left'].add_patch(c)

            if fireball_index_to_remove not in fireball_indices_to_remove:
                fireball_indices_to_remove.append(fireball_index_to_remove)
            
        return fireball_indices_to_remove, new_distance_labels_indices_to_remove
    

    def get_left_to_right_sequence_from_distance_labels(self) -> list:
        lr_sequence = ""
        skip_value = False

        for i in self.distance_labels:
            if i == 0:
                lr_sequence += str(i)
                skip_value = False
            elif i == 1:
                if skip_value:
                    skip_value = False
                else:
                    lr_sequence += str(i)
                    skip_value = True

        return lr_sequence
    

    def cyclic_slice(self, string, start, end):
        length = len(string)
        if length == 0:
            return ''

        # Adjust start and end indices to be within the length of the string
        start %= length
        end %= length

        # If start is negative, adjust it to count from the end of the string
        if start < 0:
            start += length

        # If start comes after end due to modulo, adjust end
        if start > end:
            return string[start:] + string[:end]
        else:
            return string[start:end]


    def perform_alignment(self, fireball_sequence: str) -> tuple[float, list[list[int], list[int]]]:
        """
        Attempts to align the given sequence to the de Bruijn sequence.
        
        Returns a tuple containing:
            rating: float between 0 and 1 scoring how aligned it is
            to a segment in the de Bruijn sequence
            corresponding_de_bruijn: indices of de Bruijn that correspond
            to the fireball sequence
        """
        aligner = Align.PairwiseAligner()

        aligner.match_score = 2
        aligner.mismatch_score = -3
        aligner.open_gap_score = -5
        aligner.extend_gap_score = -1

        aligner.mode = "local"

        print("Alignment Algorithm Used:", aligner.algorithm, "\n")

        sequence_length = len(fireball_sequence)
        
        # Locally align fireball with de bruijn sequence
        alignments = aligner.align(DE_BRUIJN_SEQUENCE, fireball_sequence)

        print("No. Possible Alignments:", len(alignments))
        print("Top Alignment Score:", alignments[0].score)
        alignments = sorted(alignments)
        alignment = alignments[0]
        print(alignment)

        indices = alignment.indices
        print(indices)
        start_index = indices[0][0] - indices[1][0]
        # plus one for the last fireball node
        end_index = indices[0][-1] + (sequence_length - indices[1][-1])
        print("De Bruijn Start Index:", start_index)
        print("De Bruijn End   Index:", end_index)

        de_bruijn_segment = self.cyclic_slice(DE_BRUIJN_SEQUENCE, start_index, end_index)

        # gaps found from local alignment represented with -1.
        # add these gaps into the fireball sequence for similarity analysis
        gaps = np.where(indices[1] == -1)[0]
        for gap in gaps:
            fireball_sequence = fireball_sequence[:gap] + '-' + fireball_sequence[gap:]

        print("De Bruijn Segment:", de_bruijn_segment)
        print("Fireball Sequence:", fireball_sequence)

        # Retrieve ratio of edit distance between the de bruijn segment
        # and the fireball sequence. from 0 being completely different
        # to 1 being exactly the same
        levenshtein_ratio = Levenshtein.ratio(de_bruijn_segment, fireball_sequence)
        print(f"Levenshtein Ratio: {levenshtein_ratio:.4f}\n\n")

        return levenshtein_ratio, start_index, de_bruijn_segment, fireball_sequence
    

    def consume_node(self, de_bruijn_pos: int, de_bruijn_val: str):
        node = self.fireball_nodes_queue.pop(0)
        self.fireball_points.append([node[0], node[1], de_bruijn_pos, de_bruijn_val])
        self.fireball_labels_queue.pop(0)


    def backtrack_odd_number_of_1s(self, de_bruijn_pos: int):
        """
        Every group of small gaps should have an even number of 1 values.
        In total a group of small gaps should have an odd number of nodes.
        The last node would have a big gap which would make it a 0.

        For example (going from left to right):

        `.   . . . . . . .   .`            \n
        `0   1 1 1 1 1 1 0   ?`

        6 '1s' (even) + 1 '0' = 7 (odd)


        Usually at the ends of the fireball, the algorithm does not get the
        correct number of 1s. This function is called when you assume there
        will be two 1s to label but you only label one and come across a 0
        as the next node.

        This function backtracks and labels the 1s from this point so that
        the unpaired 1 is at the beginning of the small gap group.

        NOTE: Needs more consideration for what happens in the middle of the
        fireball. Maybe count the number of '1' nodes in the small gap group
        and check distances for missing nodes?
        """

        # the current point we are going to change
        current_point = self.fireball_points[-1]
        # the position of the current point
        current_pos = len(self.fireball_points) - 1
        # how many points have we assigned a de bruijn position.
        # resets when it reaches two to start counting for the next
        # de bruijn position
        current_de_bruijn_pos_count = 0
        # the current de bruijn position we will assign to the
        # currrent node
        backtrack_de_bruijn_pos = de_bruijn_pos

        while current_pos >= 0 and current_point[3] == '1':
            # assign de bruijn position to point
            current_point[2] = backtrack_de_bruijn_pos
            current_de_bruijn_pos_count += 1

            # paired labelled, move on to previous de bruijn position
            if current_de_bruijn_pos_count == 2:
                current_de_bruijn_pos_count = 0
                backtrack_de_bruijn_pos -= 1
            
            # go to previous fireball point
            current_pos -= 1
            current_point = self.fireball_points[current_pos]


if __name__ == "__main__":
    file_path = Path(__file__)
    image_path = Path(file_path.parents[1], 'fireball_images', 'cropped', '044_2021-10-28_064629_E_DSC_0731-G_cropped.jpeg')
    fireball_071 = FireballPointPicker(image_path)
    fireball_071.show_plot()