import rawpy
import numpy as np
from scipy.fftpack import dct, idct
from skimage.util import view_as_blocks,view_as_windows
from scipy.ndimage import filters
def linear_dem(img):

    img_r = np.copy(img.raw_image)/(2**14) # Works only for E1 Camera for the moment
    img_r[img.raw_colors != 0] = 0
    img_b = np.copy(img.raw_image)/(2**14)
    img_b[img.raw_colors != 2] = 0
    img_g = np.copy(img.raw_image)/(2**14)
    img_g[(img.raw_colors != 1) & (img.raw_colors != 3)] = 0


    Kr = 0.25*np.matrix('1 2 1; 2 4 2; 1 2 1')
    Kb = Kr
    Kg = 0.25*np.matrix('0 1 0; 1 4 1; 0 1 0')

    rgb = np.zeros((img_r.shape[0], img_r.shape[1],3))
    rgb[:,:,0] = filters.convolve(img_r, Kr,mode='mirror')
    rgb[:,:,1]=  filters.convolve(img_g, Kg,mode='mirror')
    rgb[:,:,2]=  filters.convolve(img_b, Kb,mode='mirror')
    return(rgb)

def bayer_extract(RAW_path, sat_values):
        if(isinstance(RAW_path, str)):
            img = rawpy.imread(RAW_path)
        else:
            img = RAW_path

        if not isinstance(sat_values,np.ndarray):
            tmp = sat_values
            sat_values = np.zeros(4) + tmp

        G1 = img.raw_image[img.raw_colors == 0].reshape((img.raw_image.shape[0]//2, img.raw_image.shape[1]//2))/sat_values[0]
        R = img.raw_image[img.raw_colors == 1].reshape((img.raw_image.shape[0]//2, img.raw_image.shape[1]//2))/sat_values[1]
        B = img.raw_image[img.raw_colors == 2].reshape((img.raw_image.shape[0]//2, img.raw_image.shape[1]//2))/sat_values[2]
        G2 = img.raw_image[img.raw_colors == 3].reshape((img.raw_image.shape[0]//2, img.raw_image.shape[1]//2))/sat_values[3]

        RB = np.concatenate([R, B], axis = 1)
        G = np.concatenate([G1, G2], axis = 1)

        Z = np.concatenate([RB, G])

        return(Z)
def bayer_extract_spatial(im, sat_values):
    G1 = im[0::2,0::2].reshape((im.shape[0]//2, im.shape[1]//2))/sat_values[0]
    R = im[0::2,1::2].reshape((im.shape[0]//2, im.shape[1]//2))/sat_values[1]
    B = im[1::2,1::2].reshape((im.shape[0]//2, im.shape[1]//2))/sat_values[2]
    G2 = im[1::2,0::2].reshape((im.shape[0]//2, im.shape[1]//2))/sat_values[3]
    return(G1, R,B,G2)



def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

def gamma_corr(x, gamma, xmax=2**16-1):
    norm_x  = x/xmax
    luma = np.zeros_like(norm_x)
    luma[norm_x < 0.018] = gamma[1]*norm_x[norm_x < 0.018] 
    luma[norm_x >= 0.018] = 1.099*norm_x[norm_x >= 0.018]**(1/gamma[0]) - 0.099
    return(luma*xmax)

def compute_dct_domain(im_pix, c_quant):
    """
    Convert the image into DCT coefficients without performing quantization
    """
    w,h = im_pix.shape
    dct_im = np.zeros((w,h))
    for bind_i in range(w//8):
        for bind_j in range(h//8):
            im_bloc = im_pix[bind_i*8:(bind_i+1)*8,bind_j*8:(bind_j+1)*8]
            dct_im[bind_i*8:(bind_i+1)*8,bind_j*8:(bind_j+1)*8] = dct2(im_bloc-128)/(c_quant)
    return dct_im

def compute_jpeg_domain(im_pix, c_quant):
    """
    Compress the image into JPEG (simulations)
    """
    w,h = im_pix.shape
    dct_im = np.zeros((w,h))
    for bind_i in range(w//8):
        for bind_j in range(h//8):
            im_bloc = im_pix[bind_i*8:(bind_i+1)*8,bind_j*8:(bind_j+1)*8]
            dct_im[bind_i*8:(bind_i+1)*8,bind_j*8:(bind_j+1)*8] = np.round(dct2(im_bloc - 128)/(c_quant))
    dct_im = dct_im.astype(np.int32)
    return dct_im

def compute_spatial_domain(im_dct, c_quant):
    """
    DeCompress the image into greyscale (simulations)
    """
    w,h = im_dct.shape
    im_pix = np.zeros((w,h))
    for bind_i in range(w//8):
        for bind_j in range(h//8):
            im_bloc = im_dct[bind_i*8:(bind_i+1)*8,bind_j*8:(bind_j+1)*8]
            im_pix[bind_i*8:(bind_i+1)*8,bind_j*8:(bind_j+1)*8] = idct2(im_bloc * c_quant) +128
    return im_pix

def dct2(x):
    return dct(dct(x, norm='ortho').T, norm='ortho').T
    
def idct2(x):
    return idct(idct(x, norm='ortho').T, norm='ortho').T

def generate_row_scan_margin_indices(n, add_margin=False):
    interior_block = np.arange(8*8).reshape(8,8)
    corner_block = np.arange(9 *9).reshape(9,9)
    exterior_block_row = np.arange(8*9).reshape(8,9)
    exterior_block_col = np.arange(9*8).reshape(9,8)
    
    current_number = 0
    if add_margin:
        block_indices = np.zeros((8*n+2, 8*n+2), dtype=np.int)
        for i in range(0, n):
            for j in range(0,n):
                if (i == 0 or i == n-1) and (j == 0 or j == n-1):
                    start_idx_row = np.min([i*(9+8*(n-2)),9+8*(n-2)])
                    start_idx_col = np.min([j*(9+8*(n-2)),9+8*(n-2)])
                    block_indices[start_idx_row:start_idx_row+9, start_idx_col:start_idx_col+9] = current_number + corner_block
                    current_number =  current_number + 9*9
                elif (i == 0 or i == n-1):
                    #print('Exterior block {} {}'.format(i,j))
                    start_idx_row = np.min([i*(9+8*(n-2)),9+8*(n-2)])
                    start_idx_col = 9+(j-1)*8
                    #print('IDX {} {}'.format(start_idx_row,start_idx_col))
                    block_indices[start_idx_row:start_idx_row+9, start_idx_col:start_idx_col+8] = current_number + exterior_block_col
                    current_number = current_number + 9*8
                elif (j== 0 or j == n-1):
                    #print('Exterior block {} {}'.format(i,j))
                    start_idx_col = np.min([j*(9+8*(n-2)),9+8*(n-2)])
                    start_idx_row = 9+(i-1)*8
                    #print('IDX {} {}'.format(start_idx_row,start_idx_col))
                    block_indices[start_idx_row:start_idx_row+8, start_idx_col:start_idx_col+9] = current_number + exterior_block_row
                    current_number = current_number + 9*8
                else:
                    #print('Exterior block {} {}'.format(i,j))
                    start_idx_row = 9+(i-1)*8
                    start_idx_col = 9+(j-1)*8
                    #print('IDX {} {}'.format(start_idx_row,start_idx_col))
                    block_indices[start_idx_row:start_idx_row+8, start_idx_col:start_idx_col+8] = current_number + interior_block
                    current_number = current_number + 8*8
    else:
        block_indices = np.zeros(((8*n), (8*n)), dtype=np.int)
        for i in range(0, n):
            for j in range(0,n):
                    start_idx_row = (i*8)
                    start_idx_col = (j*8)
                    block_indices[start_idx_row:start_idx_row+8, start_idx_col:start_idx_col+8] = current_number + interior_block
                    current_number = current_number + 8*8

    return(np.argsort(block_indices.flatten()))
                

def remove_margin_indices(n):
    interior_block = np.arange(8*8).reshape(8,8)
    corner_block = np.arange(9*9).reshape(9,9)
    exterior_block_row = np.arange(8*9).reshape(8,9)
    exterior_block_col = np.arange(9*8).reshape(9,8)
    
    current_number = 0
    block_indices = np.zeros((8*n+2, 8*n+2), dtype=np.int)
    margin_mask = np.ones((8*n+2, 8*n+2), dtype=np.bool)
    margin_mask[:,0]  = False
    margin_mask[0,:]  = False
    margin_mask[:,-1] = False
    margin_mask[-1,:] = False
    for i in range(0, n):
        for j in range(0,n):
            if (i == 0 or i == n-1) and (j == 0 or j == n-1):
                start_idx_row = np.min([i*(9+8*(n-2)),9+8*(n-2)])
                start_idx_col = np.min([j*(9+8*(n-2)),9+8*(n-2)])
                block_indices[start_idx_row:start_idx_row+9, start_idx_col:start_idx_col+9] = current_number + corner_block
                current_number =  current_number + 9*9
            elif (i == 0 or i == n-1):
                #print('Exterior block {} {}'.format(i,j))
                start_idx_row = np.min([i*(9+8*(n-2)),9+8*(n-2)])
                start_idx_col = 9+(j-1)*8
                #print('IDX {} {}'.format(start_idx_row,start_idx_col))
                block_indices[start_idx_row:start_idx_row+9, start_idx_col:start_idx_col+8] = current_number + exterior_block_col
                current_number = current_number + 9*8
            elif (j== 0 or j == n-1):
                #print('Exterior block {} {}'.format(i,j))
                start_idx_col = np.min([j*(9+8*(n-2)),9+8*(n-2)])
                start_idx_row = 9+(i-1)*8
                #print('IDX {} {}'.format(start_idx_row,start_idx_col))
                block_indices[start_idx_row:start_idx_row+8, start_idx_col:start_idx_col+9] = current_number + exterior_block_row
                current_number = current_number + 9*8
            else:
                #print('Exterior block {} {}'.format(i,j))
                start_idx_row = 9+(i-1)*8
                start_idx_col = 9+(j-1)*8
                #print('IDX {} {}'.format(start_idx_row,start_idx_col))
                block_indices[start_idx_row:start_idx_row+8, start_idx_col:start_idx_col+8] = current_number + interior_block
                current_number = current_number + 8*8
    if n == 1:
        return(block_indices[margin_mask] +19)
    else:
        return(block_indices[margin_mask])



def block_row_scan(M,k=3, block_shift=(0,0), add_margin=False, flatten_blocks=True, reshape_as_one=False, pad_mode=0):
    """
    View of an image as macro blocks of k*8 X k*8 elements with a row scan
    k : number of block in each column/row
    block_shift : how many block to skip at the beggiing of the image, use to sample lattices 
    pad_mode : 0 = exact, no padding is done to ensure same size when shifting blocks, 1 = zero padding
    """

    (h,w) = M.shape
    #block_idx = generate_row_scan_margin_indices(k, add_margin)
    
    
    if add_margin:
        idx_start_h = 8*block_shift[0]
        idx_start_w = 8*block_shift[1]
        idx_crop_h = ((8*k)* ((h - ((8*k)+2) - idx_start_h) //(8*k)) + (8*k) +2 +  idx_start_h) 
        idx_crop_w = ((8*k)* ((w - ((8*k)+2) - idx_start_w) //(8*k)) + (8*k) +2 +  idx_start_w) 
        M_block_view = view_as_windows(M[idx_start_h:idx_crop_h,idx_start_w:idx_crop_w], (8*k + 2, 8*k+ 2), step=8*k)

    else:

        idx_start_h = 8*block_shift[0]
        idx_start_w = 8*block_shift[1]
        idx_crop_h = (8*k)* ((h - idx_start_h)//(8*k)) +  idx_start_h
        idx_crop_w = (8*k)* ((w - idx_start_w)//(8*k)) +  idx_start_w
        n_block_R = (idx_crop_h - idx_start_h)//(8*k)
        n_block_C = (idx_crop_w - idx_start_w)//(8*k)
        

        M_block_view = view_as_blocks(M[idx_start_h:idx_crop_h,idx_start_w:idx_crop_w], (8*k,8*k))

    if flatten_blocks:
        if k == 1:
            M_block_view = M_block_view.reshape(M_block_view.shape[0],M_block_view.shape[1], M_block_view.shape[2]*M_block_view.shape[3])
        else:
            block_idx = generate_row_scan_margin_indices(k, add_margin)
            M_block_view = M_block_view.reshape(M_block_view.shape[0],M_block_view.shape[1], M_block_view.shape[2]*M_block_view.shape[3])[:,:,block_idx]
    if reshape_as_one:
        M_block_view = M_block_view .reshape(M_block_view .shape[0]*M_block_view .shape[1], M_block_view .shape[2])
        
    
    return(M_block_view)

def rolling_row_scan(M,k=3, block_shift=(0,0), add_margin=False, flatten_blocks=True, stride=2):
    """
    View of an image as macro blocks of k*8 X k*8 elements with a row scan
    k : number of block in each column/row
    block_shift : how many block to skip at the beggiing of the image, use to sample lattices 
    pad_mode : 0 = exact, no padding is done to ensure same size when shifting blocks, 1 = zero padding
    """

    (h,w) = M.shape
    #block_idx = generate_row_scan_margin_indices(k, add_margin)
    
    
    if add_margin:
        idx_start_h = 8*block_shift[0]
        idx_start_w = 8*block_shift[1]
        idx_crop_h = ((8*k)* ((h - ((8*k)+2) - idx_start_h) //(8*k)) + (8*k) +2 +  idx_start_h) 
        idx_crop_w = ((8*k)* ((w - ((8*k)+2) - idx_start_w) //(8*k)) + (8*k) +2 +  idx_start_w) 
        M_roll_view = view_as_windows(M[idx_start_h:idx_crop_h,idx_start_w:idx_crop_w], (8*k + 2, 8*k+ 2), step=8*stride)

    else:

        idx_start_h = 8*block_shift[0]
        idx_start_w = 8*block_shift[1]
        idx_crop_h = (8*k)* ((h - idx_start_h)//(8*k)) +  idx_start_h
        idx_crop_w = (8*k)* ((w - idx_start_w)//(8*k)) +  idx_start_w
        n_block_R = (idx_crop_h - idx_start_h)//(8*k)
        n_block_C = (idx_crop_w - idx_start_w)//(8*k)
        

        M_roll_view = view_as_windows(M[idx_start_h:idx_crop_h,idx_start_w:idx_crop_w], (8*k,8*k), step=8*stride)

    if flatten_blocks:
        if k == 1:
            M_roll_view = M_roll_view.reshape(M_roll_view.shape[0],M_roll_view.shape[1], M_roll_view.shape[2]*M_roll_view.shape[3])
        else:
            block_idx = generate_row_scan_margin_indices(k, add_margin)
            M_roll_view = M_roll_view.reshape(M_roll_view.shape[0],M_roll_view.shape[1], M_roll_view.shape[2]*M_roll_view.shape[3])[:,:,block_idx]

        
    
    return(M_roll_view)
def lattice_to_image(L, h,w):
    
    im = np.zeros((h, w)) #h-18, w-18
    for i in range(0,3):
        for j in range(0,3):
            current_L = L[i,j][:,:,4*64:5*64]
            current_block_idx = 0
            idx_shift_x = 8*i
            idx_shift_y = 8*j
            idx_jump = 8*3
            for block_x in range(current_L.shape[0]):
                for block_y in range(current_L.shape[1]):
                    idx_start_x=idx_shift_x + (block_x*idx_jump)
                    idx_stop_x= idx_start_x + 8
                    idx_start_y= idx_shift_y + (block_y*idx_jump)
                    idx_stop_y= idx_start_y + 8
                    im[idx_start_x:idx_stop_x, idx_start_y:idx_stop_y] = current_L[block_x, block_y,:].reshape(8,8)
    return(im)

def unblock_row_scan(L,h,w):
    im = np.zeros((h, w))
    for block_x in range(L.shape[0]):
        for block_y in range(L.shape[1]):
            idx_start_x=(block_x*8)
            idx_stop_x= idx_start_x + 8
            idx_start_y= (block_y*8)
            idx_stop_y= idx_start_y + 8
            im[idx_start_x:idx_stop_x, idx_start_y:idx_stop_y] = L[block_x, block_y,:].reshape(8,8)
    return(im)

def unblock_row_scan_nb(L,nb,h,w):
    im = np.zeros((h, w))
    for block_x in range(L.shape[0]):
        for block_y in range(L.shape[1]):

            idx_start_x=(block_x*8*nb)
            idx_stop_x= idx_start_x + 8*nb
            idx_start_y= (block_y*8*nb)
            idx_stop_y= idx_start_y + 8*nb
            
            block_k = 0
            for nb_x in np.arange(nb):
                for nb_y in np.arange(nb):
                    im[idx_start_x+8*nb_x:idx_start_x+8*nb_x+8, 
                       idx_start_y+8*nb_y:idx_start_y+8*nb_y+8] = L[block_x, block_y,block_k*64:block_k*64+64].reshape(8,8)
                    block_k = block_k +1

    return(im)

def cov_to_diag_var(C):
    diag_idx = np.diag_indices(C.shape[2])
    diag_block = np.zeros((C.shape[0],C.shape[1],C.shape[2]))
    for i in np.arange(C.shape[0]):
        for j in np.arange(C.shape[1]):
            diag_block[i,j,:] = C[i,j][diag_idx]
    Variance = unblock_row_scan(diag_block, int(C.shape[0]*np.sqrt(C.shape[2])),int(C.shape[1]*np.sqrt(C.shape[2])))
    return(Variance)
   