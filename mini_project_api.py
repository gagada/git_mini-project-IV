# import Flask and jsonify
from flask import Flask, jsonify, request
# import Resource, Api and reqparser
from flask_restful import Resource, Api, reqparse
import pandas as pd
import numpy
import pickle

app = Flask(__name__)
api = Api(app)

model = pickle.load(open( "model_mini_project.p", "rb" ))

class Scoring(Resource):
    def post(self):
        json_data = request.get_json()
        df = pd.DataFrame(json_data.values(), index=json_data.keys()).transpose()
        # getting predictions from our model.
        # it is much simpler because we used pipelines during development
        res = model.predict(df)
        # we cannot send numpt array as a result
        return res.tolist() 

# assign endpoint
api.add_resource(Scoring, '/scoring')

#The last thing to do is to create an application run when the file api.py is run directly 
#(not imported as a module from another script)
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5555)