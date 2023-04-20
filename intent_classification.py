from transformers import AutoModelWithLMHead, AutoTokenizer

class IntentClassification():

    def __init__(self):

        self.tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-e2m-intent")
        self.model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-e2m-intent")

    def get_intent(self, event, max_length=16):
        self.input_text = "%s </s>" % event
        self.features = self.tokenizer([self.input_text], return_tensors='pt')

        self.output = self.model.generate(input_ids = self.features['input_ids'],
                   attention_mask=self.features['attention_mask'],
                   max_length=max_length)

        return self.tokenizer.decode(self.output[0])

        # event = "i am not feeling well"
        # print(get_intent(event))