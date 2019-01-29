import nltk
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from PIL import Image
import pickle
import matplotlib.pyplot as plt

# Maximum allowed tokens in a sentence
# Or only allowed size
word_array_size = 50
# Word vector length, we are using glove vectors which have 50 elements
word_vector_size = 50
# Location of glove vectors
glove_data = "./raw/glove.6B.50d.w2v.txt"
# Location of training data
training_data = "./raw/news.2009.en.shuffled.unique"
# Desired data set sizes
num_data = 250000

global glove_model
glove_model = KeyedVectors.load_word2vec_format(glove_data, binary=False)

# Take in a string, output a 1x50x50 numpy "word array"
def make_word_array(s):
    tokens = nltk.word_tokenize(s)
    tokens = [token.lower() for token in tokens]
    sent_length = len(tokens)
    if sent_length > word_array_size:
        raise ValueError
    else:
        word_array = []
        for token in tokens:
            nums = glove_model[token]
            list_nums = nums.tolist()
            word_array.append(list_nums)
        word_array.extend([[0] * word_vector_size] * (word_array_size - len(word_array)))
        a = np.array(word_array, dtype='float32')
        return a[np.newaxis, :], sent_length

# Take in a numpy array, output a string
def decode_word_array(a):
    sentence = []
    for vector in a[0]:
        foo = glove_model.similar_by_vector(vector, topn=1)
        bar = foo[0][1]
        # Don't decode anything with less than 0.25 magnitude (assumed to be 0-vector and padding)
        if np.linalg.norm(vector) > 0.25:
            sentence.append(foo[0][0])
        else:
            sentence.append("[pad]")
    return " ".join(sentence)

# Output a list of tokens
def decode_word_array3(a):
    sentence = []
    for vector in a[0]:
        foo = glove_model.similar_by_vector(vector, topn=1)
        bar = foo[0][1]
        # Don't decode anything with less than 0.25 magnitude (assumed to be 0-vector and padding)
        if np.linalg.norm(vector) > 0.25:
            sentence.append(foo[0][0])
        else:
            continue
            #sentence.append("[pad]")
    return sentence

# Output list of strings and cosine distances
# Used for diagnosis
def decode_word_array2(a):
    sentence = []
    for vector in a[0]:
        foo = glove_model.similar_by_vector(vector, topn=1)
        bar = foo[0]
        # don't decode anything with less than 0.25 magnitude (assumed to be 0-vector and padding)
        if np.linalg.norm(vector) > 0.25:
            sentence.append(foo[0])
        else:
            sentence.append(("[pad]",np.linalg.norm(bar[1])))
    return (sentence)

# Output list of vector magnitudes
# Used for diagnosis
def word_array_magnitudes(a):
    mags = []
    for vector in a[0]:
        mags.append(np.linalg.norm(vector))
    return (mags)

# Creates an image out of the word array
# Used for the thesis paper
def create_fancy_image(s):
    array, _ = make_word_array(s)
    array = array[0]
    scaled = (255*(array - np.min(array))/np.ptp(array)).astype('uint8')
    img = Image.fromarray(scaled)
    img = img.rotate(90).resize((400, 400))
    img.save('./out/image.png', 'png')

def preprocess():
    # Turn each sentence/line into a word array, rejecting any that are too long or out-of-vocab
    # Save each example to a file in the cache folder
    with open(training_data) as f:
        num = 0
        ls = []
        for line in f:
            try:
                word_array, sent_length = make_word_array(line)
                ls.append(sent_length)
                with open('./cache/training/true/' + str(num) + '.pickle','wb') as c:
                    pickle.dump(word_array, c)
                    num += 1
            except (KeyError, ValueError):
                pass
            if num >= num_data:
                break
    print("Average:", np.average(ls))
    print("Std:", np.std(ls))
    plt.hist(ls, bins='auto')  # arguments are passed to np.histogram
    plt.title("Histogram with 'auto' bins")
    plt.show()
    print(num)

# Main
if __name__ == "__main__":
    preprocess()
