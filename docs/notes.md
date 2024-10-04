

## CZT shapes

I think we need to just padd to 2 times the largest after zero padding.

So 

input.shape = (1,1,50,50)
output_.shape = (1,1,256, 256)

input_padded.shape = (1,1,100,100)
output_padded.shape = (1,1,512, 512)


input_padded.shape[-1] < output_padded.shape[-1]
    => pad the chirp functions associated with the input plane dimensions.

input_padded.shape[-1] > output_padded.shape[-1]
    => pad the chirp functions initialized with the output plane dimensions.


so the padding in this case is 

(512 - 100) // 2 = 206

and only for the chirp functions that are initialized with input plane dimensions.
