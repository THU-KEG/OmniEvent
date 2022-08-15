# Pretrain data processing

## 1. Raw data source

First, prepare raw data into `data_1.txt`, `data_2.txt`, ...

In the raw text files, use line breaks to separate different datas, i.e., each line has one data (for example, a long document). Notice that there may be line breaks inside data, they should be replaced with unique identifier, such as `<n>`, and replaced back in step 2.

Examples:

```eval_rst
.. code-block:: 
    :linenos:

    The Lord of the Rings<n>The Lord of the Rings is an epic high-fantasy novel by English author and scholar J. R. R. Tolkien. Set in Middle-earth, intended to be Earth at some distant time in the past, the story began as a sequel to Tolkien's 1937 children's book The Hobbit, but eventually developed into a much larger work. Written in stages between 1937 and 1949, The Lord of the Rings is one of the best-selling books ever written, with over 150 million copies sold.<n>The title refers to the story's main antagonist, the Dark Lord Sauron, who in an earlier age created the One Ring to rule the other Rings of Power given to Men, Dwarves, and Elves, in his campaign to conquer all of Middle-earth. From homely beginnings in the Shire, a hobbit land reminiscent of the English countryside, the story ranges across Middle-earth, following the quest to destroy the One Ring mainly through the eyes of the hobbits Frodo, Sam, Merry and Pippin.
    The Little Prince<n>I thought that I was rich, with a flower that was unique in all the world; and all I had was a common rose. A common rose...<n>To me, you are still nothing more than a little boy who is just like a hundred thousand other little boys. And I have no need of you. And you, on your part, have no need of me. To you, I am nothing more than a fox like a hundred thousand other foxes. But if you tame me, then we shall need each other. To me, you will be unique in all the world. To you, I shall be unique in all the world.<n>The wheat fields have nothing to say to me. And that is sad. But you have hair that is the color of gold. Think how wonderful that will be when you have tamed me! The grain, which is also golden, will bring me back the thought of you. And I shall love to listen to the wind in the wheat.
    ...
```

**Important:** Since Streaming data processing and training are used in subsequent steps, data should be pre-shuffled before putting into `data.txt` sequentially.

## 2. Tokenize

- Implement an Encoder, which gives it a line of input data and it returns you the tokenized result.

- Mapping all datas with Encoder, with the help of [multiprocessing](https://docs.python.org/3/library/multiprocessing.html)

- We also provide a tools called `indexed_dataset`, which compress the tokenized data into binary format.

There is an example code in `model_center/tools/preprocess_cpm1_lm.py`, we simplify it and made it into a example like the following:

```python
import multiprocessing
from model_center.tools import indexed_dataset

# 1. Implement an Encoder, which gives it a line of input data and it returns you the tokenized result.
class Encoder(object): 
    def initializer(self):
        Encoder.tokenizer = YourTokenizer()

    def encode(self, line):
        data = line.strip().replace("<n>", "\n") # replace back line break symbol

        doc_ids = Encoder.tokenizer.encode(data)

        max_length = 512 # model's maximum input length

        pieces = []
        while i < len(doc_ids): # split document into chunks with model's maximum length
            piece = doc_ids[i:i+max_length]
            if len(piece) < 32: # drop too short chunks
                break
            i += max_length

            pieces.append(piece)

        return pieces

if __name__ == '__main__':
    # assumes that there are 100 raw data files, named `data_1.txt` to `data_100.txt`
    for ith in range(1, 101):
        fin = open(f"data_{ith}.txt", "r", encoding="utf-8")

        # encoder use the tokenizer to encode data
        encoder = Encoder()

        # 2. Mapping all datas with Encoder, with the help of multiprocessing
        pool = multiprocessing.Pool(processes=64, initializer=encoder.initializer)
        encoded_docs = pool.imap_unordered(encoder.encode, fin, chunksize=10)

        # 3. tool `indexed_dataset` compress the tokenized data into binary format `bin_file`
        # it will also generate another small `idx_file` for saving meta information in order to decode `bin_file`.
        bin_file = os.path.join("path/to/your/tokenized_output_folder/", f"tokenized_{ith}.bin")
        idx_file = os.path.join("path/to/your/tokenized_output_folder/", f"tokenized_{ith}.idx")
    
        binary_builder = indexed_dataset.make_builder(bin_file, impl="mmap", dtype=np.int32)

        # put tokenized data into binary_builder
        for pieces in encoded_docs:
            for doc_ids in pieces:
                binary_builder.add_item(torch.IntTensor(doc_ids))

        # finish compressing tokenized data into `bin_file`, and generate meta information into `idx_file`
        binary_builder.finalize(idx_file)

        # close multiproceessing mapping
        pool.close()
```

## 3. Self-Supervised Dataset

We provide a tool `model_center.dataset.DistributedMMapIndexedDataset` for loading those compressed binary files and do extra processing,
for example, a commonly used way is to randomly mask a span and ask the model to regenerate it.

It is important to note that, tokenization is an rather slow operation, it may become a bottleneck if tokenization is perform while the model is being trained.
Thus, we put tokenization into data pre-processing stage (step 2), which is run beforehand.

However, We do not calculate things like `attention_mask` and save them into binary files because it will lead to multiplication of space occupation, 
which is not affordable for the large amount of data needed for training large language models.
Putting those extra information that are less time consuming into `__getitem__` function of [pytorch Dataset](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset),
space will be saved since the space will be recycled when the next batch arrives.

```python
from model_center.dataset import DistributedMMapIndexedDataset

class YourDataset(torch.utils.data.Dataset):
    def __init__(self, ctx : MMapIndexedDataset):
        self.ctx = ctx

    def __len__(self):
        return len(self.ctx)
    
    def __get_item_data(self, input_ids): # do extra processing
        length = input_ids.shape[0]

        # randomly mask a span in range [lef, rig)
        lef = random.randint(input_length)
        rig = random.randint(lef, input_length)

        # pretrain objective: regenerate the masked span and ignore other positions
        targets = np.full((input_length), -100) # -100 as ignore_index
        targets[lef:rig] = input_ids[lef:rig]

        # calculate attention_mask, to tell model which positions are visible
        attention_mask = (np.arange((input_length)) < lef) | (np.arange((input_length)) >= rig)
        return input_ids, targets, input_length, attention_mask

    def __getitem__(self, ith):
        input_ids = self.ctx[ith] # get the i-th data from DistributedMMapIndexedDataset
        return self.__get_item_data(ctx) # do extra processing and return

if __name__ == '__main__':
    dataset = YourDataset(
        DistributedMMapIndexedDataset("path/to/your/tokenized_output_folder", "tokenized", bmt.rank(), bmt.world_size()), 
        # the second argument "tokenized" is the common prefix of your tokenized file name,
        # here we assumes that they are called "tokenized_1.bin", "tokenized_2.idx", etc.
    )
```
