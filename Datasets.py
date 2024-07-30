class BERTDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len=256):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        question = self.dataframe.iloc[index]['question']
        answer = self.dataframe.iloc[index]['answer']

        inputs = self.tokenizer.encode_plus(
            question,
            answer,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_tensors='pt',
            truncation=True
        )

        input_ids = inputs['input_ids'].flatten()
        attention_mask = inputs['attention_mask'].flatten()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        
class T5Dataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len=256):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        question = self.data.iloc[index, 0]
        answer = self.data.iloc[index, 1]
        input_text = "question: " + question + " answer: " + answer
        inputs = self.tokenizer.encode_plus(
            input_text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=False,
            truncation=True
        )
        return {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'labels': inputs['input_ids']
        }
