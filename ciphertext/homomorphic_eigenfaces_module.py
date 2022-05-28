#AUTHORS: Henrik Tambo Buhl & Alexander Stæhr Johansen 
#CREATED: 01-05-2022

import numpy as np
import tenseal as ts
from dataclasses import dataclass
from PIL import Image

@dataclass
class EigenfacesServer:
    '''
    SUMMARY: This is the EigenfacesServer class, whose main purpose is to compute the 
    Eigenfaces algorithm on homomorphically encrypted images.
    ATTRIBUTES: The class contains four attributes: a boolean variable denoting whether the model
    has been trained, three numpy arrays, one containing the computed eigenfaces, one containing
    the mean face and one containing the projected training images.
    METHODS: The class contains an initializer, a training method for training the model, a 
    classification method for classifying an input image and thirteen utility methods supporting
    these two primary methods.
    '''
    Is_trained: bool
    eigenfaces: np.array([])
    mean_face: np.array([])
    projected_training_images: np.array([])

    def __init__(self, no_components_function, minimum_distance_function, 
    goldschmidt_initializer_function, reencrypt_function, reencrypt_vec_function, decrypt_vec_function) -> None:
        self.is_trained = False
        self.determine_components = no_components_function
        self.distance_comparison = minimum_distance_function
        self.goldschmidt_initializer = goldschmidt_initializer_function
        self.reencrypt = reencrypt_function
        self.reencrypt_vec = reencrypt_vec_function
        self.sletsenere = decrypt_vec_function

    def Train(self, normalized_training_images: np.array([]), vectorized_training_images: np.array([])) -> None:
        '''
        SUMMARY: This method follows the Eigenfaces algorithm, thus it 1) normalizes the
        training images, 2) calculates the mean face and 3) calculates the eigenfaces.
        PARAMETERS: A numpy list of training images and a numpy list of labels.
        RETURNS: None.
        '''
        # Eigenfaces procedure:
        # Step 2: Calculate the mean face, and store it in the model:
        self.mean_face = self._vector_mean(vectorized_training_images)

        # Step 3: Calculate the Eigenfaces using PCA, and store it in the model:
        self.eigenfaces = self._pca(vectorized_training_images)        

        # Step 4: Calculate the projections of the training images:
        self.projected_training_images = self._project(normalized_training_images, self.eigenfaces, self.mean_face)
        # Update the training attribute:
        self.is_trained = True

    def Classify(self, normalized_test_images: np.array([]), training_labels: np.array([])) -> np.array([]):
        '''
        SUMMARY: This method classifies an image by minimizing Euclidean distance.
        PARAMETERS: Two numpy lists, one including the image to be classified and one
        with labels of all training data.
        RETURNS: A numpy list including the classification label.
        '''
        # Make a list to store all classified labels: 
        test_labels = []
        # Project the image to be classified into the PCA-space:
        q = self._project(normalized_test_images, self.eigenfaces, self.mean_face)
        # Determine the number of test image projections: 
        n = len(q)
        # Determine the number of training image projections:
        m = len(self.projected_training_images)
        # We then calculate the distances between the input image, q, and the projections:
        for i in range(n):
            # Instantiate a list of zeros, where each entry represents the Euclidean distance
            # from the image to be classified to the nth projection:
            distances = []
            for j in range(m):
                distances.append(self._euclidean_distance(self.projected_training_images[j], q[i]))
            # We then use the client-based function that determines the index of the minimum distance:
            classification_index = self.distance_comparison(distances) #NN
            # And, return the label that corresponds to this minimum distance:
            test_labels.append(training_labels[classification_index]) #NN
        return test_labels

    def _vector_mean(self, X: np.array([])) -> np.array([]):
        '''
        SUMMARY: This method calculates the mean of a list using Goldschmidt's algorithm for division.
        PARAMETERS: A numpy list.
        RETURNS: A numpy list.
        '''
        # Initialize an empty mean variable: 
        mean = []
        # Calculate the sum of x along the vertical axis:
        __sum = np.sum(X, axis = 0)
        # Determine the number of elements in x, horizontally:
        n = len(X)
        # Calculate the dividend using Goldschmidt's method: 
        dividend = self._goldschmidt_division(1, n)
        # Determine the number of rows in x: 
        m = len(X[0])
        # Calculate the mean of x:
        for i in range(m):
            mean.append(__sum[i] * dividend)
        # Return the mean:
        return np.array(mean)

    def _goldschmidt_division(self, a, b) -> float:
        '''
        SUMMARY: This method calculates the fraction of a number, a, divided by a number, b. 
        PARAMETERS: A nominator, a, and a denominator, b.
        RETURNS: A float, a.
        '''
        #Set the number of iterations for convergence:
        no_iterations = 1
        #Calculate the initial value using the client-side function that does so: 
        r = self.goldschmidt_initializer(b)
        #Use Goldschmidt's algorithm for approximating the fraction:
        for _ in range(no_iterations):
            a = r * a
            #b = r * b
            #r = 2 + -1 * b
        #Reencrypt a using the client-side reencrypt method:
        if type(a) is ts.tensors.ckksvector.CKKSVector:
            a = self.reencrypt_vec([a])
            a = a[0]
        #Return a: 
        return a
    
    def _goldschmidt_vector_division(self, A: np.array([]), b: float) -> np.array([]):
        '''
        SUMMARY: This method divides a vector, A, with a number, b. 
        PARAMETERS: A vector nominator, A, and a denominator, b.
        RETURNS: A resultant vector, res.
        '''
        # Initialize an empty list:
        res = []
        # Store the length of the list:
        n = len(A)
        # Divide each element in the input vector, a, with a number, b:
        for i in range(0, n):
            res.append(self._goldschmidt_division(A[i], b))
        # Return the vector divided by the number: 
        return res

    def _pca(self, X: np.array([])) -> np.array([]):
        '''
        SUMMARY: This method calculates the k most significant principal components
        that capture an arbitrary %-variance in the dataset.
        PARAMETERS: A numpy list and a call to the client-side function that
        determines the number of components.
        RETURNS: A numpy list.
        '''
        # Determine the shape of the input:
        [n, d] = np.shape(X)
        # Step 1: Subtract the mean face from all the training images:
        X += -1 * self.mean_face
        # Step 2: Calculate the set of principal components:
        if(n > d):
            # Calculate the covariance matrix of X:
            C = self._matrix_mult(X.T, X)
            # Calculate the eigenvalues (Lambda) and eigenvectors (W) from the covariance matrix (C):
            Lambdas, W = self._pow_eig_comb(C)
        else:
            # Calculate the covariance matrix of X:
            C = self._matrix_mult(X, X.T)
            # Calculate the eigenvalues (Lambda) and eigenvectors (W) from the covariance matrix (C):
            Lambdas, W = self._pow_eig_comb(C)
            # And, take the dot product between the covariance matrix and the eigenvectors:
            W = self._matrix_mult(X.T, W)
            # Normalize the eigenvectors by dividing them with their norm:
            for i in range(n):
                W[:, i] = self._goldschmidt_vector_division(W[:, i], self._norm(W[:, i]))
        # Step 3: Determine the number of k components that satisfy the threshold criteria and return these:
        # Determine the number of components using the client-side function given as input: 
        k = self.determine_components(Lambdas)
        # Select the k greatest eigenvalues:
        Lambdas = Lambdas[0: k].copy()
        # And, the associated eigenvectors:
        eig_vec = []
        for i in range(0,k):
            eig_vec.append(W[i])
        # Return these:
        W = eig_vec
        return np.array(W)

    def _matrix_mult(self, A: np.array([]), B: np.array([])) -> np.array([]):
        '''
        SUMMARY: This method multiplies two matrixes A and B. 
        PARAMETERS: Two matrixes A and B.
        RETURNS: The product as a numpy array, res.
        '''
        # Determine the number of rows in both matrixes: 
        rows_a = len(A)          
        rows_b = len(B)     
        # Determine the number of coloumns in B:  
        cols_b = len(B[0])
        # Instantiate temporary product lists:
        temp_prod_1 = []
        temp_prod_2 = []
        # Compute the matrix multiplication:
        for i in range(0, rows_a):
            temp_prod_3 = []
            for j in range(0, cols_b):
                temp_prod_4 = []
                for k in range(0, rows_b):
                    temp_prod_4.append(A[i][k] * B[k][j])
                temp_prod_3.append(np.sum(temp_prod_4))
            temp_prod_2.append(temp_prod_3)
        temp_prod_1.append(self.reencrypt(temp_prod_2))
        # Return the product as a numpy array:
        res = np.array(temp_prod_1[0])
        return res

    def _pow_eig_comb(self, C: np.array([])) -> np.array([]):
        '''
        SUMMARY: This method calculates the all eigenvalues and eigenvectors, 
        from a covariance matrix.
        PARAMETERS: A numpy list, representing the covariance matrix, C.
        RETURNS: Two numpy lists: one including the eigenvalues (Lambdas) and one representing the 
        eigenvectors, W.
        '''
        # Determine the number of entries in the covariance matrix:
        n = len(C)
        # Initialize storage for the eigenvalues:
        lambdas = []
        # And, the eigenvectors:
        W = []
        # Initialize the dividend:
        dividend = 1
        # Calculate the eigenvectors:
        for i in range (n):
            # Number of iterations for approximating the eigenvectors and eigenvalues:
            no_iterations = 2 
            # Initialize the "old" eigenvector:
            w_old = 1
            # Initialize an initial vector:
            x = np.ones((n))
            #x = np.random.rand(n, 1)
            # Calculate the first eigenvector:
            w = self._goldschmidt_vector_division(x, self._norm(x))
            # Calculate all the eigenvectors:
            for j in range(no_iterations):
                # By obtaining the dot product of the covariance matrix and the first eigenvector:
                x = self._mat_vec_mult(C, w)
                # Calculate the new eigenvalue by taking the norm of the dot product:
                __lambda = self._norm(x)
                # Calculate the next eigenvector by dividing the dot product, previously calculated,
                # with the eigenvalue just calculated:
                w = self._goldschmidt_vector_division(x, __lambda)
                # If we are at the third last iteration, we break:
                if j + 2 == no_iterations:
                    # And, set the "old" eigenvector equal to the first in the current list:
                    w_old = w[0]
            # We then handle the case in which any given eigenvalue is negative:
            dividend = self._goldschmidt_division(w[0], w_old)
            # We then multiply the dividend with the eigenvalue:
            __lambda = dividend * __lambda
            # And, multiply the dividend with the eigenvector:
            w = self._vec_mult(w,dividend)
            w = self.reencrypt_vec(w)
            # And, store the current eigenvalue and eigenvectors in separate matrices:
            lambdas.append(__lambda)
            W.append(w.T)
            # And, adjust the covariance matrix:
            temp = [] 
            temp = self._vec_mult(w, -1 * __lambda)
            temp = self._vec_cross(temp, w)
            C += temp
        # And, return both the eigenvalues and eigenvectors:
        return lambdas, W

    def _mat_vec_mult(self, M: np.array([]), V: np.array([])) -> np.array([]):
        '''
        SUMMARY: This method multiplies a matrix, mat, with a vector, V.
        PARAMETERS: A matrix, M and a vector, V.
        RETURNS: The product as a numpy array, res.
        '''
        # Determine the rows and coloumns of the input matrix:
        rows_M = len(M)
        cols_M = len(M[0])
        # Instantiate temporary product lists:
        temp_prod_1 = []
        temp_prod_2 = []
        # Calculate the matrix-vector product:
        for i in range(0, rows_M):
            temp_prod_3 = []
            for j in range(0, cols_M):
                temp_prod_3.append(M[i][j] * V[j])
            temp_prod_2.append(np.sum(temp_prod_3))
        temp_prod_1.append(self.reencrypt_vec(temp_prod_2))
        # Convert the result to a numpy array:
        res = np.array(temp_prod_1[0])
        return res

    def _vec_mult(self, V: np.array([]), x: float) -> np.array([]):
        '''
        SUMMARY: This method multiples every entry in the vector, V, with a number, x:
        PARAMETERS: A vector, V, and a number, x.
        RETURNS: A resultant numpy array with each vector entry multiplied by the number.
        '''
        # Determine the number of elements in the vector:
        n = len(V)
        # Instantiate temporary product lists
        temp_prod_1 = []
        temp_prod_2 = []
        # Multiply every vector entry with the number:
        for i in range(0, n):
            temp_prod_2.append(V[i] * x)
        temp_prod_1.append(self.reencrypt_vec(temp_prod_2))
        # Convert the result to a numpy array and return it:
        res = np.array(temp_prod_1[0])
        return res

    def _vec_cross(self, V1: np.array([]), V2: np.array([])) -> np.array([]):
        '''
        SUMMARY: This method calculates cross product of two vector. The method is 
        used for shifting the covaraince matrix.
        PARAMETERS: A numpy list, representing the covariance matrix, C.
        RETURNS: Two numpy lists: one including the eigenvalues (Lambdas) and one representing the 
        eigenvectors, W.
        '''
        # Determine the number of elements in the first vector:
        n = len(V1)
        # Instantiate lists to contain temporary products:
        temp_prod_1 = []
        temp_prod_2 = []
        # Calculate the cross product:
        for i in range(0, n):
            temp_prod_3 = []
            for j in range(0, n): 
                temp_prod_3.append(V1[i] * V2[j])
            temp_prod_2.append(temp_prod_3)
        temp_prod_1.append(self.reencrypt(temp_prod_2))
        # Return the result as a numpy array:
        res = np.array(temp_prod_1[0])
        return res

    def _norm(self, X: np.array([])) -> np.array([]):
        '''
        SUMMARY: This method finds the norm, i.e length, of a vector.
        PARAMETERS: Takes the vector as input, X.
        RETURNS: Returns the the norm/length as, vector_norm.
        '''
        # Instantiate a temporary container:
        vector_norm = 0
        # Determine the number of elements in X:s
        n = len(X)
        # Raise all entries in X to the power of 2 and add them to the norm:
        for i in range(n):
            vector_norm += X[i] * X[i]
        # Find the sqrt of the sum, calculated above:
        vector_norm = self._newton_sqrt(vector_norm)
        # Return the norm.
        return vector_norm

    def _newton_sqrt(self, x0: float) -> float:
        '''
        SUMMARY: This function approximates a square root using Newtons method.
        PARAMETERS: It takes a float as input, x0.
        RETURNS: It returns a float as output, xn, which is the approximated square root of x0.
        '''
        # Declare the number of iterations to run the algorithm:
        no_iterations = 30
        # Instantiate a temporary container for the original number:
        a = x0
        # Approximate the square root:
        for _ in range(no_iterations):
            x0 = 0.5 *(x0 + self._goldschmidt_division(a, x0))
        # If a is an encrypted vector, reencrypt:
            if type(a) is ts.tensors.ckksvector.CKKSVector:
                x0 = self.reencrypt_vec([x0])
                x0 = x0[0]
        # Return the approximate square root:
        return x0

    def _euclidean_distance(self, p: np.array([]), q: np.array([])) -> np.array([]):
        '''
        SUMMARY: This method calculates the distance between two points. 
        PARAMETERS: Takes two points, encoded as numpy arrays, p and q, as input.
        OUTPUT: Returns the distance as a numpy array. 
        '''
        # We subtract:
        sub = p + -1 * q
        # Find the product:
        prod = sub * sub
        # And, calculates the vector sum of this product:
        __sum = np.sum(prod)
        # Finally, we calculate the square root of this:
        distance = self._newton_sqrt(__sum)
        # Which is the distance to be returned:
        return distance

    def _project(self, X: np.array([]), W: np.array([]), mu: np.array([])) -> np.array([]):
        '''
        SUMMARY: The following function calculates the projection of a point in a given vector space.
        PARAMETERS: A point, X, a vector space, W, and a mean, mu.
        OUTPUT: A projection, p.
        '''
        # Calculate and the return the projection:
        mu = -1 * mu
        p = []
        for mat in X:
            temp = []
            for row in mat:
                for element in row:
                    temp.append(element)
            temp = self._mat_vec_mult(W, temp + mu)
            p.append(temp)
        return p

#TODO: DOCSTRING:
@dataclass
class EigenfacesClient:
    '''
    SUMMARY: 
    ATTRIBUTES: 
    METHODS:
    '''

    #TODO: MAKE DOCSTRING:
    def __init__(self) -> None:
        self.context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree= 32768,
            coeff_mod_bit_sizes=[60, 40, 40, 40, 60]
          )
        self.context.generate_galois_keys()
        self.context.global_scale = 2**40

    def Image_preprocesser(self, images: np.array([])) -> np.array([]):
        '''
        SUMMARY: This method follows the first step in the Eigenfaces algorithm, 
        thus it 1) normalizes the training images by greyscaling them and converting
        them to a common resolution. Finally it transforms each image to its vector 
        representation.
        PARAMETERS: A numpy list of unprocessed images.
        RETURNS: A numpy list of processed images.
        '''
        #Temporary container to store processed images:
        normalized_images = []
        # Default image size declaration:
        image_default_size = [2, 2] 
        # Processing loop:
        for image in images: 
            #Convert current image to greyscale
            image = image.convert("L")
            #Resize to default size using antialiasing:
            image = image.resize(image_default_size, Image.ANTIALIAS)
            #Convert image to numpy array with data type uint8
            image = np.asarray(image, dtype=np.float64)
            #Append the processed image to temporary container:
            normalized_images.append(image)
        return normalized_images

    #TODO: DOCSTRING:
    def Image_vector_representation(self, normalized_images: np.array([])) -> np.array([]):
        '''
        SUMMARY:
        PARAMETERS:
        RETURNS: 
        '''
        #Then find the number of entries in each image:
        n_image_entries = normalized_images[0].size
        #Find the data type of each entry in the image:
        dt_images = normalized_images[0].dtype
        #Compute an empty matrix with N_image_entries number of coloumns
        #with the correct data type:
        processed_images = np.empty((0, n_image_entries), dtype=dt_images)
        #Reshape every image into a vector and stack it vertically in the
        #matrix:
        for row in normalized_images:
            processed_images = np.vstack((processed_images, np.asarray(row).reshape(1 , -1)))# 1 x r*c 
        #Return processed images:
        return processed_images

    #TODO: DOCSTRING:
    def Encrypt(self, mat: np.array([])) -> np.array([]):
        '''
        SUMMARY:
        PARAMETERS: 
        RETURNS:
        '''
        #Instantiate a temporary list:
        enc_mat = []
        #Determine the number of entries in the input matrix:
        n = len(mat)
        #Encrypt each vector in the matrix:
        for i in range(0, n):
            enc_pic = self._encrypt_vec(mat[i])           
            enc_mat.append(enc_pic)
        #Convert the list to a numpy array:
        enc_mat = np.array(enc_mat)
        #Lastly, return the encrypted matrix:
        return enc_mat

    #TODO: DOCSTRING:
    def _encrypt_vec(self, vec: np.array([])) -> np.array([]):
        '''
        SUMMARY:
        PARAMETERS: 
        RETURNS:
        '''
        #Instantiate temporary lists:
        enc_vec = []
        vec_temp = []
        #Determine the length of the input vector:
        n = len(vec)
        #Then encrypt every entry in the original vector:
        for i in range(0, n):
            vec_temp = [vec[i]]
            enc_vec.append(ts.ckks_vector(self.context, vec_temp))
        #Convert the list to a numpy array:
        enc_vec = np.array(enc_vec)
        #And, return it:
        return(enc_vec)

    #TODO: DOCSTRING AND COMMENTS:
    def _reencrypt_vec(self, vec: np.array([])) -> np.array([]):
        #Decrypt the input vector:
        dec_vec = self._decrypt_vec(vec)
        #Reencrypt the decrypted vector:
        reenc_vec = self._encrypt_vec(dec_vec)
        #Return the reencrypted vector: 
        return reenc_vec

    #TODO: DOCSTRING:
    def _reencrypt_mat(self, mat: np.array([])) -> np.array([]):
        #Decrypt the input matrix of images: 
        dec_mat = self.Decrypt(mat)
        #Reencrypt the decrypted matrix of images:
        reenc_mat = self.Encrypt(dec_mat)
        #Return the reencrypted matrix of images: 
        return reenc_mat

    #TODO: DOCSTRING:
    def Decrypt(self, mat: np.array([])) -> np.array([]):
        '''
        SUMMARY:
        PARAMETERS: 
        RETURNS:
        '''
        #Instantiate a temporary list:
        dec_mat = []
        #Determine the number of entries in the input matrix:
        n = len(mat)
        #Decrypt each vector in the matrix:
        for i in range(0, n):
            dec_pic = self._decrypt_vec(mat[i])
            dec_mat.append(dec_pic)
        #Convert decrypted matrix to a numpy array: 
        dec_mat = np.array(dec_mat)
        #Lastly, return the decrypted matrix:
        return(dec_mat)

    #TODO: DOCSTRING:
    def _decrypt_vec(self, vec: np.array([])) -> np.array([]):
        '''
        SUMMARY:
        PARAMETERS: 
        RETURNS:
        '''
        #Instantiate temporary lists:
        dec_vec = []
        vec_temp = []
        #Determine the length of the input vector:
        n = len(vec)
        #Decrypt each entry in the original vector:
        for i in range(0, n):
            vec_temp = vec[i].decrypt()
            dec_vec.append(vec_temp[0])    
        #Concert list to numpy array: 
        dec_vec = np.array(dec_vec)
        #Return the decrypted vector:       
        return(dec_vec)

    def _n_components_comparison(self, Lambdas: np.array([])) -> int: 
        '''
        SUMMARY: The following method finds the number of principal components to be used, 
        to preserve a threshold of v variance.
        PARAMETERS: The function takes a list of eigenvalues.
        RETURNS: And, returns an integer, stating how many eigenvalues to be used.
        '''
        # Variance threshold:
        v = 0.99

        # Decrypt the eigenvalues:
        Lambdas = self._decrypt_vec(Lambdas)

        # Calculate number of Eigenvalues to be used, to preserve v variance:
        for i, eigen_value_cumsum in enumerate(np.cumsum(Lambdas) / np.sum(Lambdas)):
            if eigen_value_cumsum > v:
                return i

    def _distance_comparison(self, D: np.array([])) -> int:
        '''
        SUMMARY: Finds the minimum distance in a list of distances.
        PARAMETERS: Takes a numpy list of distances as input.
        RETURNS: The minimum index in the list.
        '''
        # Decrypt the distance vector:
        D = self._decrypt_vec(D)
        return np.argmin(D) #NN

    #TODO: CORRECT DOCSTRING AND MAKE COMMENTS:
    def _goldschmidt_initializer(self, x: np.array([])) -> np.array([]):
        '''
        SUMMARY: Returns the fraction 1/x.
        PARAMETERS: Takes a denominator value, x, as input.
        RETURNS: The fraction 1/x.
        '''
        if type(x) is ts.tensors.ckksvector.CKKSVector:
            x = x.decrypt()
            x = 1 / x[0]
            x = ts.CKKSVector(self.context,[x])
            return x
        else:
            return 1 / x