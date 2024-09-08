# hangman

Tranformer model to guess the letter of hangman

Input: hang_an-> tensor of BS,max_len size, a vector of BS,vocab_size as one-hot encoding of possible/impossible candidates (impossbile candidates will be marked as 0)

Output: tensor of BS,vocab_size size, you can argmax(dim=-1) to out put the id of the guessed letter

Transformer is your usual architecture, but instead of BS,max_len,d_k output, i first convered dk to vocab_size, then removed all impossible candidates, then I multiply by a matrix that represents weights along the max_len dimension. This matrix is learned from the positional encoding and then I sum alongside max_len dimension, which lead to the output of BS,vocab_size tensor.

I used KL divergeance of loss.

Note: the input dataset file is just a word file, and the dataset will auto generate appropriate masks. I set the mask range to be 30%, because transformer is fairly accurate when more letters are guessed. But you can try remove the limit and test the result (It takes about an hour/epoch, and I don't have the time).

Accuracy for 10 epochs is around 55%
