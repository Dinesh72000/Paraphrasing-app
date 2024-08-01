from flask import Flask, request, jsonify, render_template
from transformers import T5ForConditionalGeneration,AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('t5-base')
# Define the directory where you saved the trained model
model_directory = 'model/model2'
# Load the trained model
loaded_model = T5ForConditionalGeneration.from_pretrained(model_directory)

def paraphrase(input_text):
    input_ids = tokenizer.encode(input_text, return_tensors='pt')  # Make sure to use the same tokenizer used during training

    # Generate predictions
    output = loaded_model.generate(input_ids, max_length=50, num_return_sequences=1, early_stopping=True)

    # Decode the generated output
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Print the generated text
    return generated_text

def checkSPACE(k):
    print(k)
    for i in range(len(k)-1):
        if(k[i]!=' '):
            return True
    return False

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    input_text = request.form.get('gg')  # Get the input text from the form
    arr=input_text.split('\n')
    arr1=input_text.split('\n')
    print(arr)
    for i in range(len(arr)):
        if(arr[i]!='\r'):
            arr2=arr[i].split('.')
            arr3=[]
            for j in arr2:
                if(j!='\r' and j!='' and checkSPACE(j)):
                    arr3.append(paraphrase(j)) 
            s1=''
            for j in arr3:
                s1=s1+j
            arr1[i]=s1
    s=''
    for i in arr1:
        s=s+i+'\n';  # Perform your processing here
    return render_template('index.html', prediction_text=s, input_text=input_text)


if __name__ == "__main__":
    app.run(debug=True)