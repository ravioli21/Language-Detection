from flask import Flask, request, render_template
import pickle
import preprocess

cv = pickle.load(open('cv.pkl', 'rb'))
le = pickle.load(open('labelencoder.pkl', 'rb'))

app = Flask(__name__)
labels = {1:'Danish', 2: 'Dutch', 3: 'English', 4: 'French', 5: 'German', 6: 'Greek', 7: 'Hindi', 8: 'Italian', 9: 'Kannada', 10: 'Malayalam', 11: 'Portugeese', 12: 'Russian', 13: 'Spanish'}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    model = pickle.load(open('model.pkl', 'rb'))
    if request.method == 'POST':
        txt = request.form['text']
        t_o_b = cv.transform([txt]).toarray()
        language = model.predict(t_o_b)
        corr_language = le.inverse_transform(language) 
        output = corr_language[0]
    
    return render_template('index.html', prediction='{} \n\n Language is in {}'.format(txt, output))

if __name__ == "__main__":
    app.run(debug=True)