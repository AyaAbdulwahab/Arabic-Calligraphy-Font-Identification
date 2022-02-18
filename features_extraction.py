import numpy as np
from scipy.signal import convolve2d

class Features_Extraction:

    @staticmethod
    #################################### EDM #########################################################
    def EDM_1( image):
        img = np.copy(image)
        # Padding to the img
        img= np.insert(img, 0, 1, axis=1)
        img = np.insert(img, img.shape[1], 1, axis=1)
        img= np.insert(img, 0, 1, axis=0)
        img = np.insert(img, img.shape[0], 1, axis=0)
        edm_matrix = np.zeros((3,3))
        for i in range(1,img.shape[0]-1):
            for j in range(1, img.shape[1]-1):  
                if img[i,j] == 0:
                    sub_img = img[i-1:i+2,j-1:j+2]
                    for i_sub in range(sub_img.shape[0]):
                        for j_sub in range(sub_img.shape[1]):
                            if sub_img[i_sub, j_sub] == 0:
                                edm_matrix[i_sub, j_sub] += 1

        main_arr = np.array([edm_matrix[1,2], edm_matrix[0,2], edm_matrix[0,1], edm_matrix[0,0]])
        index_arr = np.array([[0,4],[1,5],[2,6],[3,7]])[::-1]
        sort_arr = np.argsort(main_arr[::-1])[::-1]
        sorted_angles = index_arr[sort_arr].reshape(-1)
        index_map_2 = np.array([5, 2, 1, 0, 3, 6, 7, 8, 4])

        
                   #
                  ###
                 #####
                #######
               #########
              ###########   
             #############
            ###############
           #################
          ############*######
         ############*########
        #######################
       #########################
      ###########################
     #############################
    ###############################
               ########
               ########
               ########
               ########

        ###################### EDM2 ##########################################
        if edm_matrix[1,1] ==0:
            return np.zeros(13).tolist()
        edm2_matrix = np.zeros(9)
        for i in range(1,img.shape[0]-1):
            for j in range(1, img.shape[1]-1):  
                if img[i,j] == 0:
                    edm2_matrix[4] += 1
                    sub_img = img[i-1:i+2,j-1:j+2].reshape(-1)
                    sorted_angles_sub = sub_img[index_map_2[sorted_angles]]
                    first_occ = sorted_angles[np.where(sorted_angles_sub == 0)[0]]
                    edm2_matrix[index_map_2[first_occ]] += 1
        edm2_matrix = edm2_matrix.reshape((3,3))
        first_comb = np.array(edm_matrix[0,:])
        first_comb = np.append( first_comb, edm_matrix[1,2])
        first_comb = first_comb[::-1]

        edges_dir = np.max(first_comb)
        homogeneity = first_comb / np.sum(edm_matrix)
        # weight = edm_matrix[1,1]/ np.sum(original_img == 0)
        pixel_regualrity = first_comb/ edm_matrix[1,1]
        correlation = homogeneity + edm_matrix[1,1]
        edges_regularity = edm2_matrix / edm2_matrix[1,1]
        edges_regularity = edges_regularity.reshape(-1)[index_map_2][:-1]

        features_arr = edm2_matrix.reshape(-1).tolist() 
        features_arr.append([np.abs(edm_matrix[0,2] - edm_matrix[0,0])])
        return features_arr


    @staticmethod
    def lpq(img, windowSize=3):

        STFTalpha = 1 / windowSize  # alpha in STFT approaches (for Gaussian derivative alpha=1)

        img = np.float64(img)  # Convert np.image to double
        r = (windowSize - 1) / 2  # Get radius from window size
        x = np.arange(-r, r + 1)[np.newaxis]  # Form spatial coordinates in window
        
        w0 = np.ones_like(x)
        w1 = np.exp(-2 * np.pi * x * STFTalpha * 1j)
        w2 = np.conj(w1)
        
        # Get the 4 required frequencies by convolution
        convmode = 'valid'  # To leave the borders (No padding to the img)
        filterResp1 = convolve2d(convolve2d(img, w0.T, convmode), w1, convmode)
        filterResp2 = convolve2d(convolve2d(img, w1.T, convmode), w0, convmode)
        filterResp3 = convolve2d(convolve2d(img, w1.T, convmode), w1, convmode)
        filterResp4 = convolve2d(convolve2d(img, w1.T, convmode), w2, convmode)
        
        # Then get the real and imaginary part after applying the filters
        freqResp = np.dstack([filterResp1.real, filterResp1.imag,
                            filterResp2.real, filterResp2.imag,
                            filterResp3.real, filterResp3.imag,
                            filterResp4.real, filterResp4.imag])

        # Perform quantization and compute LPQ codewords
        inds = np.arange(freqResp.shape[2])[np.newaxis, np.newaxis, :]
        LPQrep = ((freqResp > 0) * (2 ** inds)).sum(2)

       
        LPQrep = np.histogram(LPQrep.flatten(), range(256))[0]
        LPQrep = LPQrep / LPQrep.sum()

        return LPQrep



    @staticmethod
    def get_lbp_hist(grayscale_img):
        size = grayscale_img.shape
        radius = 3
        LBP_matrix = np.zeros(256)
        powers_arr = np.array([0,0,0,64,0, 0,0,
                                0, 128 , 0, 0, 0, 32, 0,
                                0, 0, 0, 0, 0, 0, 0,
                                1, 0, 0, 0, 0, 0, 16,
                                0, 0, 0, 0, 0, 0, 0,
                                0, 2, 0, 0, 0, 8, 0,
                                0, 0, 0, 4, 0, 0 ,0])
        for i in range(radius,size[0]-radius):
            for j in range(radius, size[1]-radius):
                img_slice = grayscale_img[i-radius:i+radius+1, j-radius:j+radius+1]
                img_slice = (img_slice == img_slice[3,3]).reshape(-1)
                lbp = int(np.sum(img_slice * powers_arr))
                LBP_matrix[lbp] += 1
            
        return LBP_matrix