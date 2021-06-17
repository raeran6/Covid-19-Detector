from flask import Flask, render_template, request
app = Flask(__name__)
import pickle


# open a file, where you stored the pickled data
file = open('model.pkl', 'rb')
clf = pickle.load(file)
file.close()

@app.route('/', methods=["GET", "POST"])
def hello_world():
    if request.method == "POST":
        myDict = request.form
        
        fever = int(myDict['fever'])
        pain = int(myDict['pain'])
        dryCough = int(myDict['dryCough'])
        runnyNose = int(myDict['runnyNose'])
        diffBreath = int(myDict['diffBreath'])
        lossTaste = int(myDict['lossTaste'])
        chestPain = int(myDict['chestPain'])
        age = int(myDict['age'])
        
        
        # Code for inference
        inputFeatures = [fever, pain, dryCough,runnyNose,diffBreath,lossTaste,chestPain,age]
        infProb =clf.predict_proba([inputFeatures])[0][1]
        print(infProb)
        return render_template('show.html', inf=round(infProb*100))
    return render_template('index.html')
    # return 'Hello, World!' + str(infProb)


if __name__ == "__main__":
    app.run(debug=True)